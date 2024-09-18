# Numerical Methods Java Projects

This repository contains implementations of various numerical methods using Java. The projects demonstrate different techniques to solve systems of linear equations, find polynomial roots, and perform interpolation. Below is an overview of each project:

## 1. Gaussian Elimination (Project 1)
This project implements **Gaussian Elimination** to solve systems of linear equations. The algorithm transforms a given matrix into row echelon form and performs back substitution to find the solution vector.

### Features:
- Handles systems of linear equations in matrix form.
- Implements partial pivoting to increase numerical stability.

### Files:
- `GaussianElimination.java`: Contains the main logic to perform Gaussian Elimination.

---

## 2. Jacobi/Gauss-Seidel Iterative Methods (Project 2)
This project implements the **Jacobi** and **Gauss-Seidel** methods, two iterative techniques used to solve systems of linear equations. Both methods use an initial guess and iterate until a desired accuracy is reached.

### Features:
- Jacobi and Gauss-Seidel iterative solvers.
- Configurable tolerance for convergence.

### Files:
- `Jacobi.java`: Contains the implementation of the Jacobi method.
- `GaussSeidel.java`: Contains the implementation of the Gauss-Seidel method.

---

## 3. RootFinder (Project 3)
This project contains a **RootFinder** class for finding the roots of polynomials. The solution is supported by a **Polynomial** helper class, and the computations are managed by a **RootFinder Engine**.

### Features:
- Finds real and complex roots of polynomials.
- Includes a Polynomial class for defining and manipulating polynomial equations.
- Efficient root-finding algorithms.

### Files:
- `RootFinder.java`: Main class that drives the root-finding process.
- `Polynomial.java`: Helper class for representing polynomials.
- `RootFinderEngine.java`: Handles the computation logic for finding roots.

---

## 4. Interpolation (Project 4)
This project implements **Interpolation** methods to estimate values between known data points. The solution is driven by an **Interpolation Engine**.

### Features:
- Supports multiple interpolation techniques (e.g., Lagrange interpolation, Newton's divided difference interpolation).
- Flexible engine-driven structure for adding more interpolation methods.

### Files:
- `Interpolation.java`: Main class for performing interpolation.
- `InterpolationEngine.java`: Manages the interpolation computations.

---
