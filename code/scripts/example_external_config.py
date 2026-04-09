"""
Example script showing how to use TrainingConfig externally to run training.
"""

from train_treadmill_agent_jax import TrainingConfig, run_training_with_config, save_config_to_json

def main():
    # Create a custom TrainingConfig
    config = TrainingConfig(
        exp_name="external_config_example",
        seed=42,
        n_sessions=100,  # Short training for example
        num_envs=32,
        hidden_size=64,
        learning_rate=1e-4,
        entropy_weight=2.5e-3,
        critic_weight=0.05,
        gamma=0.997,
        rnn_type='GRU',
        reward_param_style=0,  # fixed
        reward_func_type=0,    # exp
        output_state_save_rate=25,
    )

    print("Running training with external config:")
    print(config)

    # Save config to JSON for reference
    save_config_to_json(config, "example_config.json")
    print("Config saved to example_config.json")

    # Run training
    final_train_state, rewards = run_training_with_config(config)

    print("
Training completed!")
    print(f"Final reward: {rewards[-1]:.4f}")

if __name__ == "__main__":
    main()