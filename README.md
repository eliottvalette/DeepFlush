# Poker AI Training System 🎲
![Poker Game Screenshot](images/GameRendered.png)

A sophisticated Texas Hold'em No-Limit poker AI training system using Monte Carlo Counterfactual Regret Minimization (MCCFR) and deep reinforcement learning.

## 🌟 Features
- **Advanced AI Architecture**
  - Transformer-based neural network for state representation
  - MCCFR for strategic decision making
  - Actor-Critic learning framework

- **Complete Poker Environment**
  - Full Texas Hold'em No-Limit rules implementation
  - 6-max table support
  - Realistic betting patterns and side pots

- **Training Capabilities**
  - Multi-agent training environment
  - Customizable simulation parameters
  - Real-time performance metrics
  - Model checkpointing and loading

- **Visualization Tools**
  - Interactive GUI for game monitoring
  - Performance metrics plotting
  - Action probability visualization

![Model Architecture](images/ModelOrganigram.png)

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.8
pytorch >= 1.7
pygame
numpy
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/poker-ai.git
cd poker-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the System

To start training:
```bash
python main.py
```

To visualize a trained model:
```bash
python main.py --rendering True
```

## 🎮 Game Configuration

The system supports various configuration options in `utils/config.py`:

```python
EPISODES = 5_000        # Number of training episodes
GAMMA = 0.9985         # Discount factor
ALPHA = 0.001          # Learning rate
MC_SIMULATIONS = 25    # MCCFR simulation count
```

## 🏗️ Architecture

### State Representation
- 142-dimensional state vector including:
  - Player cards (10 dimensions)
  - Community cards (25 dimensions)
  - Hand information (12 dimensions)
  - Game phase (5 dimensions)
  - Stack information (6 dimensions)
  - And more...

### Model Structure
![Training Process](images/training_process.png)

1. **Input Projection Layer**
   - Projects 142D state to 64D embedding

2. **Transformer Encoder**
   - Multi-head attention mechanism
   - Position-wise feed-forward networks

3. **Dual Output Heads**
   - Action probabilities (policy)
   - State value estimation (critic)

## 📊 Performance

The system demonstrates strong performance in:
- Strategic decision making
- Pot odds calculation
- Position-based play
- Hand strength evaluation

![Performance Metrics](images/Poker_metrics.jpg)

## 🛠️ Advanced Usage

### Custom Agent Training

```python
agent = PokerAgent(
    state_size=STATE_SIZE,
    device=device,
    action_size=12,
    gamma=GAMMA,
    learning_rate=ALPHA,
    load_model=False
)
```

### Shared Model Training

```python
shared_model = agent_list[0].model
for agent in agent_list:
    agent.model = shared_model
```

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{poker_ai_2024,
  author = {Eliott},
  title = {Poker AI Training System},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/eliottvalette/DeepFlush.git}
}
```

## 🙏 Acknowledgments

- OpenAI for transformer architecture inspiration
- DeepMind for MCCFR implementation insights
- The poker AI research community