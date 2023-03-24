![[iris-gray.png|center|200]]
# Introducing the Iris Pie  



---

## Team

* MirekB 

* LjubicaT

* SergeyP

* AlenaE (Brno Institute of Technology)


---


* Current status

* Live example

* Little devils

* Vox populi on some design choices 

* Mid-sized elephant in the room: Packaging, installation, dependencies


---

## Progress


## Things in working order

**Structural models**

* Preparser, parser

* Algorithmic differentiator (quasi-symbolic)

* First-order solution of stationary and nonstationary models

* First-order simulation


**Data management**

* Dates

* Time series

* Databank

* Import/export from/to CSV

* Some bits of plotting (wrapper around `plotly`)


---


## Things being worked on


**Structural models**

* Nonlinear steady state solver

* Simulation plans for simulation inversion (exo/endo)


**Reporting**

* Connecting Iris Pie to rephrase.js


---


## Things ahead of us

**Structural models**

* Nonlinear dynamic solver


**Time series models**

* Time series models (VARs et al) – leveraging the linear system features
  developed for structural models


----


## Next milestone

End May

* Steady state of nonlinear models

* First-order simulations of nonlinear models

* Simulation plans

* Advanced plotting and reporting


----


## Little devils

Plenty of syntactical differences – some of them may become a bit devilish
if you're not aware of them

Matlab | Python | Comment
---|---|---
`m1=m` | `m1=m.copy()` | Variables are "pointers"
`:` outside brackets | `:` only when indexing | Poor design choice
`1:n` | `range(0, n)` | Zero based, left open
`func` or `func()` | `func()` | Functions are first-class citizens
`obj=func(obj, ...)` | `obj.func(...)` | More than just syntactical difference


---

## Vox populi

* Syntactic distinction btw anticipated vs unanticipated shocks

* Default options: linear, flat/growth, anticipate, 

* Option/object/function names

* 

---

## Setup

* Installing packages and dependencies requires command line

* Choice of a "standard-issue" IDE


