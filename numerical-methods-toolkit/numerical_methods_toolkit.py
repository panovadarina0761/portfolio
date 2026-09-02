import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import interp1d, CubicSpline


def newton_method(f, df, x0, tolerance=1e-8, max_iter=100):
    """Solve f(x)=0 with Newton's method and return iteration history."""
    history = []
    x = float(x0)
    for iteration in range(max_iter + 1):
        fx = f(x)
        history.append((iteration, x, fx))
        if abs(fx) <= tolerance:
            return x, history, True
        dfx = df(x)
        if abs(dfx) < 1e-14:
            return x, history, False
        x = x - fx / dfx
    return x, history, False


def relaxation_method(f, x0, tau=0.01, tolerance=1e-8, max_iter=5000):
    """Fixed-step relaxation x_(k+1)=x_k-tau*f(x_k)."""
    history = []
    x = float(x0)
    for iteration in range(max_iter):
        fx = f(x)
        history.append((iteration, x, fx))
        x_next = x - tau * fx
        if abs(x_next - x) <= tolerance:
            history.append((iteration + 1, x_next, f(x_next)))
            return x_next, history, True
        x = x_next
    return x, history, False


def gaussian_elimination_pivoting(A, b):
    """Solve Ax=b with Gaussian elimination and partial pivoting."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    aug = np.hstack([A, b.reshape(-1, 1)])
    for col in range(n):
        pivot = col + np.argmax(np.abs(aug[col:, col]))
        if abs(aug[pivot, col]) < 1e-14:
            raise ValueError("Matrix is singular or nearly singular.")
        if pivot != col:
            aug[[col, pivot]] = aug[[pivot, col]]
        for row in range(col + 1, n):
            factor = aug[row, col] / aug[col, col]
            aug[row, col:] -= factor * aug[col, col:]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        rhs = aug[i, -1] - np.dot(aug[i, i + 1:n], x[i + 1:n])
        x[i] = rhs / aug[i, i]
    return x


def sor_method(A, b, omega=1.25, tolerance=1e-8, max_iter=10000, x0=None):
    """Solve Ax=b using Successive Over-Relaxation."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    history = []
    for iteration in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            sigma1 = np.dot(A[i, :i], x[:i])
            sigma2 = np.dot(A[i, i + 1:], x_old[i + 1:])
            raw = (b[i] - sigma1 - sigma2) / A[i, i]
            x[i] = (1 - omega) * x_old[i] + omega * raw
        residual = np.linalg.norm(b - A @ x)
        history.append((iteration, residual))
        if residual <= tolerance:
            return x, history, True
    return x, history, False


def lagrange_value(x_nodes, y_nodes, x_value):
    """Evaluate the Lagrange interpolation polynomial."""
    result = 0.0
    for i in range(len(x_nodes)):
        term = y_nodes[i]
        for j in range(len(x_nodes)):
            if i != j:
                term *= (x_value - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += term
    return result


def hermite_value(x_nodes, y_nodes, derivatives, x_value):
    """Evaluate the Hermite interpolation polynomial."""
    n = len(x_nodes)
    result = 0.0
    for i in range(n):
        li = 1.0
        derivative_sum = 0.0
        for j in range(n):
            if i != j:
                li *= (x_value - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
                derivative_sum += 1.0 / (x_nodes[i] - x_nodes[j])
        hi = li**2 * (1 - 2 * (x_value - x_nodes[i]) * derivative_sum)
        ki = (x_value - x_nodes[i]) * li**2
        result += y_nodes[i] * hi + derivatives[i] * ki
    return result


PI = np.pi


def newton_demo_function(x):
    return 3 * x**2 - np.cos(PI * x) ** 2


def newton_demo_derivative(x):
    return 6 * x + 2 * PI * np.cos(PI * x) * np.sin(PI * x)


def relaxation_demo_function(x):
    return x**3 - 10 * x**2 + 44 * x + 29


def interpolation_demo_function(x, k=0.1, m=3):
    return np.exp(k * x) - m * np.arctan(-x**3)


def interpolation_demo_derivative(x, k=0.1, m=3):
    return k * np.exp(k * x) + m * 3 * x**2 / (1 + x**6)


class NumericalMethodsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Numerical Methods Toolkit")
        self.root.geometry("1180x720")
        self.root.minsize(960, 620)

        # Use the available screen space. On Windows this opens the app maximized.
        try:
            self.root.state("zoomed")
        except tk.TclError:
            pass

        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")

        header = ttk.Frame(root, padding=(18, 14))
        header.pack(fill=tk.X)
        ttk.Label(header, text="Numerical Methods Toolkit", font=("Segoe UI", 20, "bold")).pack(anchor=tk.W)
        ttk.Label(header, text="Interactive exploration of root finding, linear systems and interpolation methods").pack(anchor=tk.W, pady=(2, 0))

        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))
        self.root_tab = ttk.Frame(notebook)
        self.linear_tab = ttk.Frame(notebook)
        self.interp_tab = ttk.Frame(notebook)
        notebook.add(self.root_tab, text="Root Finding")
        notebook.add(self.linear_tab, text="Linear Systems")
        notebook.add(self.interp_tab, text="Interpolation")

        self._build_root_tab()
        self._build_linear_tab()
        self._build_interp_tab()

    def _build_root_tab(self):
        left = ttk.Frame(self.root_tab, padding=14); left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(self.root_tab, padding=10); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        ttk.Label(left, text="Method", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        self.root_method = ttk.Combobox(left, state="readonly", values=["Newton Method", "Relaxation Method"], width=28)
        self.root_method.current(0); self.root_method.pack(fill=tk.X, pady=(4, 12))
        ttk.Label(left, text="Initial guess").pack(anchor=tk.W)
        self.root_x0 = ttk.Entry(left); self.root_x0.insert(0, "0.5"); self.root_x0.pack(fill=tk.X, pady=(4, 10))
        ttk.Label(left, text="Tolerance").pack(anchor=tk.W)
        self.root_tol = ttk.Entry(left); self.root_tol.insert(0, "1e-8"); self.root_tol.pack(fill=tk.X, pady=(4, 10))
        ttk.Label(left, text="Relaxation parameter tau").pack(anchor=tk.W)
        self.root_tau = ttk.Entry(left); self.root_tau.insert(0, "0.01"); self.root_tau.pack(fill=tk.X, pady=(4, 12))
        ttk.Button(left, text="Run Method", command=self.run_root_method).pack(fill=tk.X)
        ttk.Separator(left).pack(fill=tk.X, pady=10)
        ttk.Label(left, text="Result", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        self.root_result = tk.Text(left, width=34, height=12, wrap=tk.WORD); self.root_result.pack(fill=tk.BOTH, pady=(4, 0))
        self.root_fig = Figure(figsize=(7.0, 4.8), dpi=100)
        self.root_ax = self.root_fig.add_subplot(111)
        self.root_canvas = FigureCanvasTkAgg(self.root_fig, master=right); self.root_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def run_root_method(self):
        try:
            x0 = float(self.root_x0.get()); tol = float(self.root_tol.get()); method = self.root_method.get()
            if method == "Newton Method":
                root, history, converged = newton_method(newton_demo_function, newton_demo_derivative, x0, tol)
                title = "Newton Method Convergence"
            else:
                tau = float(self.root_tau.get())
                root, history, converged = relaxation_method(relaxation_demo_function, x0, tau, tol)
                title = "Relaxation Method Convergence"
            self.root_result.delete("1.0", tk.END)
            self.root_result.insert(tk.END, f"Converged: {'Yes' if converged else 'No'}\nApproximate root: {root:.10f}\nIterations: {max(0, len(history)-1)}\nFinal |f(x)|: {abs(history[-1][2]):.3e}")
            it = [r[0] for r in history]; err = [max(abs(r[2]), 1e-16) for r in history]
            self.root_ax.clear(); self.root_ax.semilogy(it, err, marker="o")
            self.root_ax.set_title(title); self.root_ax.set_xlabel("Iteration"); self.root_ax.set_ylabel("|f(x)|"); self.root_ax.grid(True, alpha=0.3)
            self.root_canvas.draw()
        except Exception as exc:
            messagebox.showerror("Input error", str(exc))

    def _build_linear_tab(self):
        left = ttk.Frame(self.linear_tab, padding=14); left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(self.linear_tab, padding=10); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        ttk.Label(left, text="Method", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        self.linear_method = ttk.Combobox(left, state="readonly", values=["Gaussian Elimination with Pivoting", "Successive Over-Relaxation"], width=32)
        self.linear_method.current(0); self.linear_method.pack(fill=tk.X, pady=(4, 12))
        ttk.Label(left, text="Relaxation parameter omega").pack(anchor=tk.W)
        self.omega_entry = ttk.Entry(left); self.omega_entry.insert(0, "1.25"); self.omega_entry.pack(fill=tk.X, pady=(4, 10))
        ttk.Label(left, text="Tolerance").pack(anchor=tk.W)
        self.linear_tol = ttk.Entry(left); self.linear_tol.insert(0, "1e-8"); self.linear_tol.pack(fill=tk.X, pady=(4, 12))
        ttk.Button(left, text="Solve System", command=self.run_linear_method).pack(fill=tk.X)
        ttk.Separator(left).pack(fill=tk.X, pady=10)
        ttk.Label(left, text="Result", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        self.linear_result = tk.Text(left, width=36, height=20, wrap=tk.WORD); self.linear_result.pack(fill=tk.BOTH, pady=(4, 0))
        self.linear_fig = Figure(figsize=(7.0, 4.8), dpi=100)
        self.linear_ax = self.linear_fig.add_subplot(111)
        self.linear_canvas = FigureCanvasTkAgg(self.linear_fig, master=right); self.linear_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    @staticmethod
    def demo_linear_system():
        A = np.array([[10.,-1.,2.,0.,0.,0.],[-1.,11.,-1.,3.,0.,0.],[2.,-1.,10.,-1.,0.,0.],[0.,3.,-1.,8.,-1.,1.],[0.,0.,0.,-1.,10.,-2.],[0.,0.,0.,0.,-2.,10.]])
        def f(t): return t**7 - 2*t**2 + 3
        grid = np.linspace(0, 1, 1001)
        b = np.array([np.trapz(f(grid) * grid**(i-1), grid) for i in range(1, 7)])
        return A, b

    def run_linear_method(self):
        try:
            A, b = self.demo_linear_system(); method = self.linear_method.get()
            self.linear_result.delete("1.0", tk.END); self.linear_ax.clear()
            if method == "Gaussian Elimination with Pivoting":
                x = gaussian_elimination_pivoting(A, b)
                residual = np.linalg.norm(b - A @ x)
                self.linear_result.insert(tk.END, "Solution vector:\n" + "\n".join(f"x{i+1} = {v:.8f}" for i,v in enumerate(x)) + f"\n\nResidual norm: {residual:.3e}\nDeterminant: {np.linalg.det(A):.6f}\nCondition number: {np.linalg.cond(A):.6f}")
                self.linear_ax.bar(np.arange(1, len(x)+1), x); self.linear_ax.set_title("Solution Vector"); self.linear_ax.set_xlabel("Variable"); self.linear_ax.set_ylabel("Value"); self.linear_ax.grid(True, axis="y", alpha=0.3)
            else:
                omega = float(self.omega_entry.get()); tol = float(self.linear_tol.get())
                x, history, converged = sor_method(A, b, omega, tol)
                self.linear_result.insert(tk.END, f"Converged: {'Yes' if converged else 'No'}\nIterations: {len(history)}\n\nSolution vector:\n" + "\n".join(f"x{i+1} = {v:.8f}" for i,v in enumerate(x)) + f"\n\nFinal residual: {history[-1][1]:.3e}")
                self.linear_ax.semilogy([r[0] for r in history], [max(r[1],1e-16) for r in history]); self.linear_ax.set_title("SOR Residual Convergence"); self.linear_ax.set_xlabel("Iteration"); self.linear_ax.set_ylabel("Residual norm"); self.linear_ax.grid(True, alpha=0.3)
            self.linear_canvas.draw()
        except Exception as exc:
            messagebox.showerror("Computation error", str(exc))

    def _build_interp_tab(self):
        left = ttk.Frame(self.interp_tab, padding=14); left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(self.interp_tab, padding=10); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        ttk.Label(left, text="Interpolation comparison", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(left, text="k parameter").pack(anchor=tk.W, pady=(12,0))
        self.k_entry = ttk.Entry(left); self.k_entry.insert(0, "0.1"); self.k_entry.pack(fill=tk.X, pady=(4,10))
        ttk.Label(left, text="m parameter").pack(anchor=tk.W)
        self.m_entry = ttk.Entry(left); self.m_entry.insert(0, "3"); self.m_entry.pack(fill=tk.X, pady=(4,12))
        ttk.Button(left, text="Build Comparison", command=self.run_interpolation).pack(fill=tk.X)
        ttk.Separator(left).pack(fill=tk.X, pady=10)
        ttk.Label(left, text="Error summary", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        self.interp_result = tk.Text(left, width=38, height=18, wrap=tk.WORD); self.interp_result.pack(fill=tk.BOTH, pady=(4,0))
        self.interp_fig = Figure(figsize=(7.0, 4.8), dpi=100)
        self.interp_ax = self.interp_fig.add_subplot(111)
        self.interp_canvas = FigureCanvasTkAgg(self.interp_fig, master=right); self.interp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def run_interpolation(self):
        try:
            k = float(self.k_entry.get()); m = float(self.m_entry.get())
            x_nodes = np.array([-1 + 0.6*i for i in range(6)], dtype=float)
            y_nodes = interpolation_demo_function(x_nodes, k, m)
            derivatives = interpolation_demo_derivative(x_nodes, k, m)
            x_dense = np.linspace(-1, 2, 400)
            true_values = interpolation_demo_function(x_dense, k, m)
            lag = np.array([lagrange_value(x_nodes, y_nodes, v) for v in x_dense])
            her = np.array([hermite_value(x_nodes, y_nodes, derivatives, v) for v in x_dense])
            lin = interp1d(x_nodes, y_nodes, kind="linear")(x_dense)
            cub = CubicSpline(x_nodes, y_nodes)(x_dense)
            errors = {"Lagrange": np.max(np.abs(lag-true_values)), "Hermite": np.max(np.abs(her-true_values)), "Linear spline": np.max(np.abs(lin-true_values)), "Cubic spline": np.max(np.abs(cub-true_values))}
            self.interp_result.delete("1.0", tk.END); self.interp_result.insert(tk.END, "Maximum absolute error:\n\n" + "\n".join(f"{n}: {v:.3e}" for n,v in errors.items()))
            self.interp_ax.clear(); self.interp_ax.plot(x_dense, true_values, label="Original function", linewidth=2); self.interp_ax.plot(x_dense, lag, label="Lagrange"); self.interp_ax.plot(x_dense, her, label="Hermite"); self.interp_ax.plot(x_dense, cub, label="Cubic spline"); self.interp_ax.scatter(x_nodes, y_nodes, label="Interpolation nodes", zorder=4)
            self.interp_ax.set_title("Interpolation Method Comparison"); self.interp_ax.set_xlabel("x"); self.interp_ax.set_ylabel("y"); self.interp_ax.grid(True, alpha=0.3); self.interp_ax.legend()
            self.interp_canvas.draw()
        except Exception as exc:
            messagebox.showerror("Input error", str(exc))


def main():
    root = tk.Tk()
    NumericalMethodsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
