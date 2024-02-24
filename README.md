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
| ![subcricical](docs/img/fhn-phase-1.png) | ![supercritical](docs/img/fhn-phase-2.png) |


In a special choice of hyperparameters of $a = b = 0$ one obtains the equations of the Van der Pol circuit. This can be constructed using electronic components. This motivates us to seek ways to create human-engineered electronic versions of other threshold effect dynamical systems.

One exotic component of interest is the memristor. Similar to the transistor's journey of being hypothesised in the 1920s, and built many years later in the 1940s, the memristor was hypothesised in 1971 and first built in 2008. Fang et al (2022) fork the FHN model, modifying the Van der Pol circuit to use a memristor instead of a non-linear resistor. This uses much less power! They successfully demonstrate threshold logic ALUs in their paper.

A cooler, supercooled even, component is the Josephson Junction (JJ), which uses the Josephson Effect from quantum mechanics, predicted in 1962. Chalkiadakis and Hizanidis (2022) demonstrate that coupling two JJs together creates a 'neuron' which exhibits a threshold effect. These neurons operate one hundred million times faster than a neuron in a human brain! The only problem is we do not yet know how to create 'synapses' (connections between neurons) for JJ neurons, since the threshold effect is exhibited by the relative phase difference of the JJs rather than an output current.

| Josephson Junction Neuron, with $\vec{w} \cdot \vec{x}$ supplied by $I_{\text{in}}$, and $b$ supplied by $I_{\text{b}}$ |
| - |
| ![Josephson Junction Neuron](docs/img/jjn-from-paper.png) |

To be clear, whilst JJs use a quantum mechanical effect, we could use them in a conventional-computing setting with threshold logic, as well as in quantum-computing which has been done by D-Wave Systems. If synapses were to be discovered, mass-manufacturing of JJs becomes popularised, and cloud computing companies safely house low-temperature supercooled JJ computers for ssh-ing into, **we could be looking at threshold logic Josephson Junction computing overthrowing digital logic transistor computing for the next generational leap of computational power**.

## libThresholdLogic.ExampleNetworks

### Adders

#### HalfAdder

#### FullAdder

#### GenericBitAdder

#### GenericNumberAdder

### LogicGates

#### NOT - a humble inverter, often not required

#### AND and NAND, OR and NOR - splitting the plane



#### AND and NOR, NAND and OR - a new regrouping

#### GAND and GNAND - generalised bit selection

`x == b` and `x != b`

#### HammingGate - The ultimate $\{0, 1\}^n$ linear gate



#### XOR and XNOR - Non-linearly separable gates

### Multipliers

#### BitMultiplier2x2

#### GenericBitMultiplier
