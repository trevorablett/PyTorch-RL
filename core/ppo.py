import torch
import matplotlib.pyplot as plt


def ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, optim_value_iternum, states, actions,
             returns, advantages, fixed_log_probs, clip_epsilon, l2_reg, aux_states=None, expert_states=None,
             expert_actions=None, expert_aux_states=None, expert_loss_weight=0.1):

    """update critic"""
    # TODO fairly sure that the output of the critic should be matching normalized returns, not unnormalized

    if expert_states is not None:
        if aux_states is not None:
            exp_pred_act, _, _ = policy_net(expert_states, expert_aux_states)
        else:
            exp_pred_act, _, _ = policy_net(expert_states)
        bc_surr = expert_loss_weight * (exp_pred_act - expert_actions).pow(2).sum(1).mean()

    if states is not None:  # could happen in the case in correctional setting where only expert data provided
        for _ in range(optim_value_iternum):
            if aux_states is not None:
                values_pred = value_net(states, aux_states)
            else:
                values_pred = value_net(states)
            value_loss = (values_pred - returns).pow(2).mean()
            # weight decay
            for param in value_net.parameters():
                value_loss += param.pow(2).sum() * l2_reg
            optimizer_value.zero_grad()
            value_loss.backward()
            optimizer_value.step()

        """update policy"""
        if aux_states is not None:
            log_probs = policy_net.get_log_prob(states, actions, aux_states)
        else:
            log_probs = policy_net.get_log_prob(states, actions)

        ratio = torch.exp(log_probs - fixed_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages

        policy_surr = -torch.min(surr1, surr2).mean()
        if expert_states is not None:
            policy_surr += bc_surr
    else:
        assert expert_states is not None, "Must have either expert or non-expert states/actions, not neither"
        policy_surr = bc_surr

    import ipdb; ipdb.set_trace()

    optimizer_policy.zero_grad()
    policy_surr.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
    optimizer_policy.step()