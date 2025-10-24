import numpy as np
import matplotlib.pyplot as plt
import argparse


def compute_hc2(T, Tc, xi0_m, d_sc_m):
    # Physical constant: superconducting flux quantum (Weber)
    phi0 = 2.067833848e-15

    # Avoid domain issues at/above Tc
    red = np.clip(1.0 - T / Tc, 0.0, None)

    mu0Hc2_perp = phi0 / (2.0 * np.pi * xi0_m**2) * red
    mu0Hc2_para = (
        phi0 * np.sqrt(12.0) / (2.0 * np.pi * xi0_m * d_sc_m) * np.sqrt(red)
    )
    return mu0Hc2_perp, mu0Hc2_para


def main():
    parser = argparse.ArgumentParser(
        description=(
            "根据GL公式生成 μ0Hc2 vs T 曲线 (⊥ 与 ∥) 并保存图片。\n"
            "公式: μ0Hc2^⊥ = Φ0/[2πξ_GL(0)^2]*(1-T/Tc); "
            "μ0Hc2^∥ = Φ0√12/[2πξ_GL(0)d_SC]*√(1-T/Tc)"
        )
    )
    parser.add_argument("--Tc", type=float, default=10.0, help="临界温度 Tc [K], 默认 10 K")
    parser.add_argument(
        "--xi0_nm",
        type=float,
        default=5.0,
        help="ξ_GL(0) [nm], 默认 5 nm",
    )
    parser.add_argument(
        "--d_nm",
        type=float,
        default=20.0,
        help="超导薄膜厚度 d_SC [nm], 默认 20 nm",
    )
    parser.add_argument(
        "--nT",
        type=int,
        default=400,
        help="温度采样点数，默认 400",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default=None,
        help="输出图片文件名",
    )
    args = parser.parse_args()

    Tc = args.Tc
    xi0_m = args.xi0_nm * 1e-9
    d_sc_m = args.d_nm * 1e-9

    # 温度从 0 到接近 Tc (避免 T = Tc 时的计算问题)
    T = np.linspace(0.0, Tc * 0.999, args.nT)

    mu0Hc2_perp, mu0Hc2_para = compute_hc2(T, Tc, xi0_m, d_sc_m)

    plt.figure(figsize=(6, 4))
    plt.plot(T, mu0Hc2_perp, label=r"$\mu_0 H_{c2}^{\perp}(T)$")
    plt.plot(T, mu0Hc2_para, label=r"$\mu_0 H_{c2}^{\parallel}(T)$")
    plt.xlabel("T (K)")
    plt.ylabel(r"$\mu_0 H_{c2}$ (T)")
    plt.title("$\mu_0 H_{c2}$ vs T")
    plt.legend()
    plt.grid(True, ls=":", alpha=0.6)
    plt.tight_layout()
    if args.outfile:
        plt.savefig(args.outfile, dpi=200)
        print(f"图像已保存: {args.outfile}")
        return
    else:
        print("打开交互式窗口显示图像（关闭窗口以结束程序）…")
        plt.show()
        return



if __name__ == "__main__":
    main()
