# Maestro Simulator Framework

A high-quality, modular reinforcement learning-based task scheduling simulator for IoT networks with energy constraints.

## Architecture Overview

The Maestro framework implements a complete S-A-R-S' (State-Action-Reward-State) reinforcement learning loop for task scheduling in energy-constrained IoT networks. The architecture emphasizes:

- **High Cohesion**: Each module has a single, well-defined responsibility
- **Low Coupling**: Modules depend on abstractions, not concrete implementations
- **Design Patterns**: Extensive use of Abstract Base Classes (ABC) for extensibility
- **Clean Code**: Following PEP 8 and best practices throughout

### Core Modules

#### 1. Configuration Module (`config.py`)

Centralized management of all simulation parameters.

**Key Features:**
- Dataclass-based configuration with validation
- Clear separation of concerns (network, task, timing, RL parameters)
- Extensible parameter structure

```python
config = SimulationConfig(
    NUM_NODES=25,
    NUM_PRIORITY_LEVELS=3,
    SAMPLING_PERIOD=1.0,
    SIMULATION_DURATION=3600.0,
)
SimulationConfig.validate(config)  # Validate all parameters
```

#### 2. Task Queue Module (`task_queue.py`)

Multi-level priority queue implementation for task management.

**Key Features:**
- K-level priority queues with FIFO ordering within each level
- Automatic tracking of enqueue/dequeue times
- Queue statistics and monitoring
- Abstract `ITaskQueue` interface for alternative implementations

**State Exposed to RL:**
- Queue length (sum of all K queues)
- Highest priority in queue
- Maximum wait time (head task of highest priority queue)

```python
queue = TaskQueue(num_priority_levels=3, priority_levels=[1, 2, 3])
queue.enqueue(task)
task = queue.peek()
total_length = queue.get_total_length()
max_wait = queue.get_max_waiting_time()
```

#### 3. Task Manager Module (`task_manager.py`)

Task generation and type management using Poisson process.

**Key Features:**
- Task type definitions with configurable attributes
- Poisson process task generation (exponential inter-arrival times)
- Dynamic task attribute modification (for RL)
- Task type statistics

**Task Type Attributes:**
- Priority: [1, K]
- Max delay: Maximum acceptable latency
- Required nodes: Number of nodes needed for execution
- Energy threshold: Minimum energy level required

```python
task_manager = TaskManager(
    task_types=[...],
    lambda_arrival_rate=0.1,
    num_nodes=25,
)
task, next_arrival_time = task_manager.generate_next_task(current_time)
task_manager.update_task_type_attribute("sampling", "priority", 2)
```

#### 4. State Calculator Module (`state_calculator.py`)

Converts raw sensor data into normalized MDP state vectors.

**State Vector (6-dimensional):**
$$s_t = [E_{avg}, \sigma_E^2, P_{in}, Q_{len}, Q_{pri}, T_{wait}]$$

- **$E_{avg}$**: Average normalized network energy [0, 1]
- **$\sigma_E^2$**: Network energy variance
- **$P_{in}$**: Energy collection rate (V/s), normalized [-1, 1]
- **$Q_{len}$**: Task queue length, normalized [0, 1]
- **$Q_{pri}$**: Highest priority in queue, normalized
- **$T_{wait}$**: Max waiting time, normalized [0, 1]

**Key Features:**
- Energy history buffering
- Automatic normalization of all state components
- Energy rate calculation (delta E / delta T)
- Statistics collection for debugging

```python
calculator = StateCalculator(num_nodes=25)
calculator.update_energy_data(voltages, timestamp)
state = calculator.compute_state_vector(queue_length, priority, wait_time)
```

#### 5. Simulink Control Interface Module (`simulink_interface.py`)

Handles communication with MATLAB/Simulink simulation environment.

**Key Features:**
- Periodic sampling and simulation control
- Mock mode for testing without MATLAB
- Energy reading and task activation
- MATLAB engine abstraction

**Main Operations:**
1. Initialize Simulink environment
2. Set pause times for periodic sampling
3. Read current simulation time
4. Sample node energy levels
5. Activate tasks on selected nodes
6. Dynamic sampling period adjustment

```python
interface = SimulinkInterface(sampling_period=1.0, num_nodes=25)
interface.initialize()
energies = interface.get_node_energies()
interface.activate_task_on_nodes(node_ids=[0, 1, 2], threshold=2.5)
interface.set_sampling_period(2.0)  # Called by RL agent
```

#### 6. RL Interface Module (`rl_interface.py`)

Abstract interface for reinforcement learning integration.

**Design Philosophy:**
- Algorithm-agnostic: Pluggable RL implementations
- Predefined action and reward structures
- No specific RL algorithm implementation (CQL can be plugged in)

**Action Space (4-dimensional):**
- Sampling period adjustment
- Task priority modification
- Energy threshold adjustment
- Required nodes modification

**Reward Design:**
- Task completion bonus: +1.0 per completed task
- Energy efficiency bonus: Based on consumption rate
- Response time penalty: Proportional to average wait time
- Deadline miss penalty: -10.0 per missed deadline

```python
# Interface initialization (agent to be implemented)
rl_interface = RLInterface(
    state_dimension=6,
    action_dimension=4,
    rl_agent=None,  # Pluggable agent
    enabled=False
)

# Selecting action
action = rl_interface.select_action(state)

# Computing reward
reward = rl_interface.compute_reward(
    tasks_completed=5,
    energy_used=100.0,
    avg_response_time=15.0,
    deadlines_missed=0,
    time_step=1.0
)

# Updating agent
rl_interface.update_agent(state, action, reward, next_state, done=False)
```

#### 7. Main Simulator Module (`simulator.py`)

Orchestrates all components in complete S-A-R-S' simulation loop.

**Simulation Flow:**
1. **Initialization Phase**
   - Set up all modules
   - Start Simulink interface
   - Generate initial tasks
   - Sample initial energy data

2. **Main Loop** (repeating until `SIMULATION_DURATION`):
   - Update simulation time
   - Sample energy levels
   - Check for task arrivals
   - Compute current state
   - Get RL action
   - Apply RL action
   - Process executable tasks
   - Compute reward and update agent

3. **Cleanup Phase**
   - Stop Simulink interface
   - Save trajectory data
   - Log final metrics

```python
config = SimulationConfig(SIMULATION_DURATION=3600.0)
simulator = MaestroSimulator(config, rl_enabled=False)
simulator.run()
```

## Usage Examples

### Basic Usage

```python
from src.config import SimulationConfig
from src.simulator import MaestroSimulator

# Create configuration
config = SimulationConfig()

# Create and run simulator
simulator = MaestroSimulator(config, rl_enabled=False)
simulator.run()
```

### With Custom Configuration

```python
config = SimulationConfig(
    NUM_NODES=50,
    NUM_PRIORITY_LEVELS=4,
    SAMPLING_PERIOD=0.5,
    SIMULATION_DURATION=7200.0,
    LAMBDA_ARRIVAL_RATE=0.2,
)

simulator = MaestroSimulator(config)
simulator.run()
```

### With RL Agent (Future)

```python
from src.rl_interface import IRLAgent

class CQLAgent(IRLAgent):
    """Implement Conservative Q-Learning agent"""
    def get_action(self, state):
        # TODO: Implement CQL action selection
        pass
    
    def update(self, state, action, reward, next_state, done):
        # TODO: Implement CQL update
        pass

cql_agent = CQLAgent()
rl_interface = RLInterface(rl_agent=cql_agent, enabled=True)

simulator = MaestroSimulator(config, rl_enabled=True)
simulator.run()
```

## Design Patterns Used

### 1. Abstract Base Class (ABC) Pattern
- `ITaskQueue`: Abstract task queue interface
- `ITaskManager`: Abstract task manager interface
- `IStateCalculator`: Abstract state calculator interface
- `ISimulinkInterface`: Abstract Simulink interface
- `IRLAgent`: Abstract RL agent interface

This allows:
- Easy swapping of implementations
- Clear contracts between modules
- Testing with mock implementations

### 2. Dependency Injection
- Modules receive dependencies as constructor arguments
- Loose coupling between components
- Easy testing and configuration

### 3. Configuration Object Pattern
- Centralized configuration management
- Validation and default values
- Easy parameter modification

### 4. Strategy Pattern
- Different RL algorithms can be plugged in
- Action selection strategies
- Reward computation strategies

## Code Quality Standards

### High Cohesion
- Each module has a single responsibility
- Related functionality is grouped together
- Clear, focused class methods

### Low Coupling
- Dependency injection for all dependencies
- Abstract interfaces for external dependencies
- Minimal inter-module communication
- Modules can be tested independently

### Extensibility
- Abstract base classes for key components
- Clear extension points for new functionality
- Pluggable RL agent interface
- Configurable parameters

### Documentation
- Comprehensive module docstrings
- Clear function/method documentation
- Type hints throughout
- Inline comments for complex logic

### Testing
- Mock mode for testing without MATLAB
- Configurable random seeds
- Trajectory data collection for analysis
- Comprehensive logging

## RL Interface - Implementation Guide

To implement a new RL agent (e.g., CQL):

1. Create a class inheriting from `IRLAgent`
2. Implement required methods:
   - `get_action(state)`: Select action based on state
   - `update(state, action, reward, next_state, done)`: Update agent
   - `save(path)`: Save model
   - `load(path)`: Load model

3. Instantiate and plug into `RLInterface`

```python
class MyRLAgent(IRLAgent):
    def get_action(self, state: np.ndarray) -> RLAction:
        # Implement your algorithm
        action_vector = self.policy_network(state)
        return RLAction.from_vector(action_vector)
    
    def update(self, state, action, reward, next_state, done):
        # Implement your learning algorithm
        pass
    
    def save(self, path):
        # Save model weights
        pass
    
    def load(self, path):
        # Load model weights
        pass

# Use in simulator
agent = MyRLAgent()
rl_interface = RLInterface(rl_agent=agent, enabled=True)
```

## Configuration Parameters

### Network Configuration
- `NUM_NODES`: Total nodes in network (default: 25)
- `VOLTAGE_MAX`: Maximum capacitor voltage (default: 4.3V)

### Task Configuration
- `NUM_PRIORITY_LEVELS`: Number of priority levels K (default: 3)
- `NUM_TASK_TYPES`: Number of task types M (default: 3)
- `LAMBDA_ARRIVAL_RATE`: Poisson process parameter (default: 0.1)

### Simulation Configuration
- `SAMPLING_PERIOD`: Simulink sampling period (default: 1.0s)
- `SIMULATION_DURATION`: Total simulation time (default: 3600.0s)

### Energy Configuration
- `ENERGY_THRESHOLD_MIN`: Minimum energy threshold (default: 1.8V)
- `ENERGY_THRESHOLD_MAX`: Maximum energy threshold (default: 3.8V)

### RL Configuration
- `RL_ACTION_DIM`: Action space dimension (default: 4)
- `ENABLE_LOGGING`: Enable detailed logging (default: True)
- `SAVE_TRAJECTORY`: Save trajectory data (default: True)

## File Structure

```
maestro/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── task_queue.py          # Priority queue implementation
│   ├── task_manager.py        # Task generation and management
│   ├── state_calculator.py    # State vector computation
│   ├── simulink_interface.py  # MATLAB/Simulink communication
│   ├── rl_interface.py        # RL abstraction layer
│   └── simulator.py           # Main simulator orchestrator
├── tests/
│   ├── test_task_queue.py
│   ├── test_task_manager.py
│   ├── test_state_calculator.py
│   └── test_simulator.py
├── example_run.py             # Example usage
├── requirements.txt
└── README.md
```

## Logging

All modules use Python's standard `logging` module. Configure logging level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Considerations

- **Mock Mode**: Runs without MATLAB, suitable for fast testing
- **Energy Buffering**: Energy history kept in memory (configurable size)
- **Lazy Computation**: State only computed when needed
- **Efficient Queuing**: O(1) enqueue/dequeue operations

## Future Enhancements

- [ ] Implement CQL algorithm
- [ ] Implement other RL algorithms (DQN, A3C)
- [ ] Real MATLAB/Simulink integration
- [ ] Distributed simulation support
- [ ] Performance optimization
- [ ] Visualization tools
- [ ] Result analysis toolkit
