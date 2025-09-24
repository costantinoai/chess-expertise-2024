from manim import *
import matplotlib.pyplot as plt
import numpy as np

class BarPlotsGridAndTransform(Scene):
    def construct(self):
        # -----------------------------
        # 1) CREATE FIRST BAR PLOT
        # -----------------------------
        x1 = np.arange(3)
        y1 = [3, 5, 2]

        fig1, ax1 = plt.subplots()
        ax1.bar(x1, y1, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ax1.set_title("Bar Plot 1")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")

        barplot_1 = MatplotlibMobject(fig1)
        barplot_1.scale(0.5)  # scale down so it fits nicely

        # -----------------------------
        # 2) CREATE SECOND BAR PLOT
        # -----------------------------
        x2 = np.arange(5)
        y2 = [1, 2, 3, 4, 5]

        fig2, ax2 = plt.subplots()
        ax2.bar(x2, y2, color="skyblue")
        ax2.set_title("Bar Plot 2")
        ax2.set_xlabel("Categories")
        ax2.set_ylabel("Values")

        barplot_2 = MatplotlibMobject(fig2)
        barplot_2.scale(0.5)

        # -----------------------------
        # 3) POSITION IN A 2x1 GRID (TOP/BOTTOM)
        # -----------------------------
        # By default, we can just put the first barplot up and the second down
        barplot_1.to_edge(UP)
        barplot_2.to_edge(DOWN)

        self.add(barplot_1, barplot_2)
        self.wait(1)

        # -----------------------------
        # 4) SUPERIMPOSE THE TWO PLOTS
        # -----------------------------
        # Animate them to the center so they overlap
        self.play(
            barplot_1.animate.move_to(ORIGIN),
            barplot_2.animate.move_to(ORIGIN)
        )
        self.wait(1)

        # -----------------------------
        # 5) CREATE THIRD BAR PLOT (TARGET)
        # -----------------------------
        x3 = np.linspace(0, 2, 4)
        y3 = [2, 4, 1, 3]

        fig3, ax3 = plt.subplots()
        ax3.bar(x3, y3, color=["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"])
        ax3.set_title("Bar Plot 3 (Target)")
        ax3.set_xlabel("New X")
        ax3.set_ylabel("New Y")

        barplot_3 = MatplotlibMobject(fig3)
        barplot_3.scale(0.5).move_to(ORIGIN)

        # -----------------------------
        # 6) TRANSFORM THE FIRST TWO PLOTS INTO THE THIRD
        # -----------------------------
        # A direct Transform between an entire group and a single figure
        # can look abrupt. FadeTransform is often smoother:
        group_12 = VGroup(barplot_1, barplot_2)
        self.play(FadeTransform(group_12, barplot_3))
        self.wait(1)
