# Testing-Linear-Separability-between-Two-Sets-in-any-dimension-python

This project is the **Python reference implementation** of the linear separability testing algorithm proposed in:

Shuiming Zhong and Huan Lyu, "A New Sufficient & Necessary Condition for Testing Linear Separability between Two Sets", IEEE TPAMI, 2024.

Functionality: Given two finite point sets (A, B \subset \mathbb{R}^n), the code can determine whether they are linearly separable (LS / NLS) and outputs a “degree of linear separability” (LS_Degree).

A MATLAB implementation of the algorithm has already been released at
“[https://github.com/lhfbest/Testing-Linear-Separability-between-Two-Sets-in-any-dimension”](https://github.com/lhfbest/Testing-Linear-Separability-between-Two-Sets-in-any-dimension”).
To make it more convenient to use in a Python environment, this repository provides the corresponding Python implementation. The two implementations are functionally equivalent.

---

## 1. Citation

If you use this algorithm or this code in a paper or project, please cite:

> Zhong S, Lyu H, Lu X, Wang B, Wang D. A New Sufficient & Necessary Condition for Testing Linear Separability between Two Sets. *IEEE Transactions on Pattern Analysis and Machine Intelligence*. 2024 Jan 22;PP. doi: 10.1109/TPAMI.2024.3356661.

---

## 2. Environment Dependencies

* **Python** ≥ 3.10 (uses the `|` union type annotation syntax) 
* **NumPy**
* **Matplotlib** (only needed for plotting, i.e., when running `demo_and_draw.py`) 

---

Okay, below is a **concise Chinese description version**, which only explains how to use the demo and, in one sentence, mentions the function of each `.py` file. It does not use code blocks at all, so you can paste it directly into the README.

---

3. Recommended first use
   Run `demo.py`/`demo_and_draw.py`.

   * The program will automatically generate two point sets A and B (their linear separability is controlled by the value of `mode`).
   * It calls the core algorithm to determine whether these two classes of points are linearly separable.
   * It prints to the screen:

     * Linear separability (LS means linearly separable, NLS means not linearly separable),
     * The LS_Degree value (indicating the degree of linear separability),
     * Computation time.
     * A visualization (This is only available when you use demo_and_draw.py, and only for dimensions n = 2 or 3).


Academic communication and collaboration are welcome.

The author’s main research interests include:
* Embodied intelligence (currently)
* 3D vision (currently)
* World models (currently)
* Machine learning
* Classification problems
* Clustering problems

* Contact 1: **[12551016@zju.edu.cn](mailto:12551016@zju.edu.cn)**

* Contact 2: **[1726341330@qq.com](mailto:1726341330@qq.com)**

* Contact 3: **[huanlyu@nuist.edu.cn](mailto:huanlyu@nuist.edu.cn)** (may not be checked)

---

## 8. Time Information

* Creation date: 2025-11-16
* Last modified date: 2025-11-16

```
