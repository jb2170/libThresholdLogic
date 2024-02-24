# libThresholdLogic

A Python library for creating threshold logic Neuron Networks in a declarative way. This library is based primarily on my [MSc Dissertation](Dissertation.pdf) work, and is an improved refactoring of the codebase. One is encouraged to read the dissertation, however a very abridged summary is given below, followed by documentation for the codebase and its example networks.

## Installation

- Clone the repository
- `python -m venv venv`
- `source ./venv/bin/activate`
- `pip install -e .`

Tests executable scripts are available in the `test` folder which correspond to the networks in `src/libThresholdLogic/ExampleNetworks`

## Threshold Logic

### TLDR

The primary component of threshold logic is the (*classical) Perceptron, whose scalar output $y$ for input $\vec{x} \in \{0, 1\}^n$, weights $\vec{w} \in \R^n$, and bias $b \in \R$ is determined by

$$
y = H(\vec{w} \cdot \vec{x} - b)
$$

where $H$ is the Heaviside step function. (*in modern usage a perceptron employs a smoother transfer function in place of $H$ such as the sigmoid, for suitability within calculus)

With clever choices for $\vec{w}$ and $b$, over which one has freedom, one can create logic gates and ALUs. $\vec{x}$ can be of arbitrary dimension allowing a high degree of connectivity when linking perceptrons' inputs and outputs with each other. This is one of the big advantages over digital (binary) logic, leading to significantly reduced (often $\mathcal{O}(n) \to \mathcal{O}(\log(n))$) component-count in common ALUs.

### A quick rundown of why we are interested in threshold logic

*nemo igitur vir magnus sine aliquo adflatu divino umquam fuit* - Cicero

Threshold activation is exhibited by biological neurons in nature. Scientific experimentation observing how neurons and synapses react under electrical currents procured their mathematical modelling as dynamical systems of differential equations. One such model is the two Fitzhugh-Nagumo (FHN) equations (1961), simplifying the four differential equations of earlier work done by Hodgkin and Huxley (1952).

$$
\dot{x} = c(y + x - x^3 / 3 + z) \\
\dot{y} = -(x - a + by) / c
$$

Bifurcation theory analysis of the FHN equations reveals a neuron reaching its threshold for activation to be equivalent to a limit-cycle dynamical system reaching its first bifurcation orbit. "The location of the singular point
P and hence its stability depends on z" (Fitzhugh, 1961:450).

| $z = 0.0$, subcricical; orbit-trapped to nullcline-intersection | $z = -0.4$, supercritical; first orbit |
| - | - |
| ![subcritical](docs/img/fhn-phase-1.png) | ![supercritical](docs/img/fhn-phase-2.png) |


In a special choice of hyperparameters of $a = b = 0$ one obtains the equations of the Van der Pol circuit. This can be constructed using electronic components. This motivates us to seek ways to create human-engineered electronic versions of other threshold effect dynamical systems.

One exotic component of interest is the memristor. Similar to the transistor's journey of being hypothesised in the 1920s, and built many years later in the 1940s, the memristor was hypothesised in 1971 and first built in 2008. Fang et al (2022) fork the FHN model, modifying the Van der Pol circuit to use a memristor instead of a non-linear resistor. This uses much less power! They successfully demonstrate threshold logic ALUs in their paper.

A cooler, supercooled even, component is the Josephson Junction (JJ), which uses the Josephson Effect from quantum mechanics, predicted in 1962. Chalkiadakis and Hizanidis (2022) demonstrate that coupling two JJs together creates a 'neuron' which exhibits a threshold effect. These neurons operate one hundred million times faster than a neuron in a human brain! The only problem is we do not yet know how to create 'synapses' (connections between neurons) for JJ neurons, since the threshold effect is exhibited by the relative phase difference of the JJs rather than an output current.

| Josephson Junction Neuron, with $\vec{w} \cdot \vec{x}$ supplied by $I_{\text{in}}$, and $b$ supplied by $I_{\text{b}}$ |
| - |
| ![Josephson Junction Neuron](docs/img/jjn-from-paper.png) |

To be clear, whilst JJs use a quantum mechanical effect, we could use them in a conventional-computing setting with threshold logic, as well as in quantum-computing which has been done by D-Wave Systems. If synapses were to be discovered, mass-manufacturing of JJs becomes popularised, and cloud computing companies safely house low-temperature supercooled JJ computers for ssh-ing into, **we could be looking at threshold logic Josephson Junction computing overthrowing digital logic transistor computing for the next generational leap of computational power**.

### The codebase

We focus on 'synthetic' threshold logic in this codebase, that is the lossless mathematical derivations and functions, independent of a particular technology such as Josephson Junctions or Memristors. This provides as much portability as possible. Each file within the codebase has good further documentation.

Every network / ALU inherits from the `NeuronNetwork` class and overrides its abstract `__init__` method, within it creating and connecting the network's neurons. This way of declaratively constructing a network, say `my_network`, means that one can peacefully call `my_network(inputs)` without having to think about the evaluation order of the neurons; the `NeuronNetwork.__call__` method is coded to do that, a depth-first algorithm. All one has to provide to `super().__init__` is an input layer of `ProxyNeuron`s and an output layer of neurons, corresponding to the IO of the network. A `ProxyNeuron` is a wrapper class for a `BaseNeuron` component which one intends to provide at a later time. For example one may evaluate the network on its own, which connects `ConstNeuron`s to the input layer, or one may connect networks together, chaining IO. Think of a `ProxyNeuron` like a bare wire sticking out of a 555 timer chip.

## libThresholdLogic.ExampleNetworks

I would definitely recommend [Ben Eater][ben-eater-yt]'s YouTube channel for learning about how computers work at the lowest level.

All of the following networks except `HammingGate` (which came to me recently) are described in the thesis as well. We make use of the term little/big 'bittian' (inspired by endian-ness) when talking about bitwise binary enumeration of numbers.

### `Adders.py`

A simple operation we could ask of a computer is to add numbers together. We start with bitwise addition.

#### `HalfAdder`

Adds two binary bits together, yielding two binary outputs, a 'sum' (1s) bit and a 'carry' (2s) bit.

Observe how pleasant the way of connecting neurons together is.

```py
class HalfAdder(NeuronNetwork):
    def __init__(self) -> None:
        neuron_sum = Perceptron(1.0)
        neuron_carry = Perceptron(1.0)

        neurons = [neuron_sum, neuron_carry]

        # inhibitory connection from carry to sum
        neuron_sum.add_input(-2.0, neuron_carry)

        input_layer = [ProxyNeuron() for _ in range(2)]

        # excitatory connections from inputs
        for neuron_src in input_layer:
            neuron_sum.add_input(1.0, neuron_src)
            neuron_carry.add_input(0.5, neuron_src)

        output_layer = neurons

        super().__init__(input_layer, output_layer)
```

With a corresponding neuron diagram

![alt](docs/img/svg/Half-Adder.svg.png)

#### `FullAdder`

It would be nice to be able to chain half-adders together for adding two k-bit numbers. The adders would experience a ripple-effect, carrying the carry bits forwards to the next adder each time we add some bits. For this we would need to connect another input to the adder's neurons, which is indeed what we do for a `FullAdder`.

A full adder takes in three inputs, and provides two outputs.

![alt](docs/img/svg/Full-Adder.svg.png)

#### `GenericBitAdder`

XXX TODO Complete README

#### `GenericNumberAdder`

### `LogicGates.py`

#### `NOT` - a humble inverter, often not required

#### `AND` and `NAND`, `OR` and `NOR` - splitting the plane



#### `AND` and `NOR`, `NAND` and `OR` - a new regrouping

#### `GAND` and `GNAND` - generalised bit selection

`x == b` and `x != b`

#### `HammingGate` - The ultimate $\{0, 1\}^n$ mono-neuron gate



#### `XOR` and `XNOR` - Non-linearly separable gates

### `Multipliers.py`

#### `BitMultiplier2x2`

#### `GenericBitMultiplier`



[ben-eater-yt]: https://www.youtube.com/watch?v=dXdoim96v5A
