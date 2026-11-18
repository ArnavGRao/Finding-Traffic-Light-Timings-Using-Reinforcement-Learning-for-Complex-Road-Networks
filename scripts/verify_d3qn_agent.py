import os
import tempfile
import numpy as np
import torch

from src.agents import D3QNAgent, DuelingQNetwork, PrioritizedReplayBuffer


def test_dueling_network():
    batch_size, state_dim, num_actions = 32, 33, 4
    net = DuelingQNetwork(state_dim=state_dim, num_actions=num_actions, hidden_dim=128)
    states = torch.randn(batch_size, state_dim)

    value, advantage = net.forward_streams(states)
    q_values = net(states)

    assert value.shape == (batch_size, 1), f"Expected {(batch_size, 1)}, got {value.shape}"
    assert advantage.shape == (batch_size, num_actions), f"Expected {(batch_size, num_actions)}, got {advantage.shape}"
    assert q_values.shape == (batch_size, num_actions), f"Expected {(batch_size, num_actions)}, got {q_values.shape}"

    # Verify Q(s, a) == V(s) + (A(s, a) - mean(A))
    expected_q = value + (advantage - advantage.mean(dim=-1, keepdim=True))
    assert torch.allclose(q_values, expected_q, atol=1e-6), "Dueling aggregation mismatch"

    # Verify mean(Q) == V(s)
    assert torch.allclose(q_values.mean(dim=-1, keepdim=True), value, atol=1e-6), "Mean Q != V(s)"

    # Verify gradients flow through both V and A streams
    loss = q_values.sum()
    loss.backward()
    for name, param in net.named_parameters():
        assert param.grad is not None, f"Missing gradient for {name}"
    print("[PASS] DuelingQNetwork forward, stream centering, and backward gradients.")


def test_prioritized_replay_buffer():
    np.random.seed(42)
    buf = PrioritizedReplayBuffer(capacity=16, state_dim=33, num_actions=4, alpha=0.6, beta_start=0.4)

    for i in range(16):
        s = np.full(33, float(i), dtype=np.float32)
        ns = np.full(33, float(i + 1), dtype=np.float32)
        mask = np.array([0.0, -1e9, -1e9, -1e9], dtype=np.float32)
        buf.add(s, action=i % 4, reward=float(i), next_state=ns, done=(i == 15), mask=mask, next_mask=mask)

    assert len(buf) == 16

    # Boost priority of index 0 massively and check that it is sampled disproportionately
    buf.update_priorities(np.array([15]), np.array([100.0]))  # tree_idx 15 == data_idx 0 for capacity 16
    batch = buf.sample(batch_size=16)

    assert batch["states"].shape == (16, 33)
    assert batch["next_masks"].shape == (16, 4)
    assert np.all(batch["is_weights"] > 0.0) and np.all(batch["is_weights"] <= 1.0 + 1e-6)
    zero_hits = int(np.sum(batch["states"][:, 0] == 0.0))
    assert zero_hits > 5, f"Expected high-priority item 0 to dominate sample, got {zero_hits}/16"
    print(f"[PASS] PrioritizedReplayBuffer sampling & IS weights (high-priority hits={zero_hits}/32).")


def test_masked_double_dqn_target():
    agent = D3QNAgent(state_dim=33, num_actions=4, gamma=0.9, dueling=True, double_dqn=True, device="cpu")
    batch_size = 8

    states = torch.randn(batch_size, 33)
    actions = torch.zeros(batch_size, dtype=torch.int64)
    rewards = torch.ones(batch_size, dtype=torch.float32) * 2.0
    next_states = torch.randn(batch_size, 33)
    dones = torch.zeros(batch_size, dtype=torch.float32)

    # Only action 2 is valid in next_state; actions 0, 1, 3 are blocked (-1e9)
    next_masks = torch.full((batch_size, 4), -1e9, dtype=torch.float32)
    next_masks[:, 2] = 0.0

    # Force online_net to assign huge unmasked Q-values to blocked action 0
    with torch.no_grad():
        for p in agent.online_net.parameters():
            p.zero_()
        agent.online_net.advantage_stream[-1].bias.copy_(torch.tensor([999.0, 50.0, 1.0, 10.0]))

        # Set target_net to output known distinct Q-values: [100, 200, 5.0, 300]
        for p in agent.target_net.parameters():
            p.zero_()
        agent.target_net.advantage_stream[-1].bias.copy_(torch.tensor([100.0, 200.0, 5.0, 300.0]))
        # Mean of [100, 200, 5, 300] = 151.25 -> centered Q[2] = 5.0 - 151.25 = -146.25
        expected_q_a2 = agent.target_net(next_states)[:, 2]

    is_weights = torch.ones(batch_size, dtype=torch.float32)
    loss, td_errors, y_t = agent.compute_target_and_loss(
        states, actions, rewards, next_states, dones, next_masks, is_weights
    )

    expected_y_t = rewards + 0.9 * expected_q_a2
    assert torch.allclose(y_t, expected_y_t, atol=1e-5), f"Target mismatch: {y_t} vs {expected_y_t}"
    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss became NaN/Inf under -1e9 mask"

    # Verify act() respects mask even when action 0 has Q=999
    chosen = agent.act(np.zeros(33, dtype=np.float32), next_masks[0].numpy(), epsilon=0.5)
    assert chosen == 2, f"Expected masked act() to select only valid action 2, got {chosen}"
    print("[PASS] Masked Double-DQN target Y_t and valid-action selection.")


def test_training_convergence_and_checkpoint():
    np.random.seed(7)
    torch.manual_seed(7)
    agent = D3QNAgent(
        state_dim=33,
        num_actions=4,
        batch_size=32,
        buffer_capacity=512,
        lr=1e-3,
        dueling=True,
        double_dqn=True,
    )

    # Populate synthetic transitions where action == int(state[0] > 0) gives reward +1.0 else -1.0
    for _ in range(256):
        s = np.random.randn(33).astype(np.float32)
        ns = np.random.randn(33).astype(np.float32)
        mask = np.zeros(4, dtype=np.float32)
        mask[2:] = -1e9  # Actions 0 and 1 valid
        a = agent.act(s, mask, epsilon=0.5)
        optimal_a = 1 if s[0] > 0 else 0
        r = 1.0 if a == optimal_a else -1.0
        agent.store_transition(s, a, r, ns, done=True, mask=mask, next_mask=mask)

    losses = []
    for _ in range(150):
        metrics = agent.train_step()
        assert metrics is not None
        losses.append(metrics["loss"])

    assert np.mean(losses[-20:]) < np.mean(losses[:20]), (
        f"Expected loss decrease, got start={np.mean(losses[:20]):.4f} end={np.mean(losses[-20:]):.4f}"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "d3qn_test.pt")
        agent.save(ckpt_path)
        reloaded = D3QNAgent(state_dim=33, num_actions=4)
        reloaded.load(ckpt_path)
        assert reloaded.train_steps == 150
        for p1, p2 in zip(agent.online_net.parameters(), reloaded.online_net.parameters()):
            assert torch.equal(p1, p2), "Checkpoint parameter mismatch"

    print(
        f"[PASS] Synthetic training convergence (loss {np.mean(losses[:20]):.4f} -> {np.mean(losses[-20:]):.4f}) & checkpointing."
    )


if __name__ == "__main__":
    test_dueling_network()
    test_prioritized_replay_buffer()
    test_masked_double_dqn_target()
    test_training_convergence_and_checkpoint()
    print("\nAll D3QN agent unit tests passed.")