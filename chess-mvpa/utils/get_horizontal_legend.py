import matplotlib.pyplot as plt

fig, ax = plt.subplots()

groups = [
    ("Early Visual", "#a6cee3"),
    ("Intermediate Visual", "#1f78b4"),
    ("Sensorimotor", "#b2df8a"),
    ("Auditory", "#33a02c"),
    ("Temporal", "#fb9a99"),
    ("Posterior", "#e31a1c"),
    ("Anterior", "#fdbf6f"),
]

# Layout settings
legend_y = 1.05              # y-position in axis coords
box_width = 0.035            # wider color patch
box_height = 0.04
text_pad = 2                 # smaller gap between box and label (pixels)
item_pad = 12                # spacing between items (pixels)

# Initialize renderer
fig.canvas.draw()
renderer = fig.canvas.get_renderer()

# Start x-position in axis coords
x_cursor = 0.05
dpi = fig.dpi

for i, (name, color) in enumerate(groups):
    # Draw color patch
    ax.add_patch(
        plt.Rectangle(
            (x_cursor, legend_y - box_height / 2),
            box_width,
            box_height,
            color=color,
            transform=ax.transAxes,
            clip_on=False
        )
    )

    # Estimate text width in pixels
    text = ax.text(0, 0, name, fontsize=14, transform=None)
    bbox = text.get_window_extent(renderer=renderer)
    text_width_px = bbox.width
    text.remove()

    # Convert pixel width to axis coords
    text_width_ax = ax.transAxes.inverted().transform((text_width_px, 0))[0] - ax.transAxes.inverted().transform((0, 0))[0]

    # Text position (convert text_pad to axis coords)
    text_offset_ax = text_pad / dpi
    text_x = x_cursor + box_width + text_offset_ax

    ax.text(
        text_x,
        legend_y,
        name,
        fontsize=14,
        va='center',
        ha='left',
        transform=ax.transAxes,
    )

    # Update x_cursor for next item (skip item_pad after last)
    if i < len(groups) - 1:
        item_pad_ax = item_pad / dpi
    else:
        item_pad_ax = 0

    x_cursor = text_x + text_width_ax + item_pad_ax

# Draw surrounding legend box
legend_x0 = 0.05 - 0.01
legend_x1 = x_cursor + 0.01
legend_y0 = legend_y - box_height / 2 - 0.01
legend_y1 = legend_y + box_height / 2 + 0.01

ax.add_patch(
    plt.Rectangle(
        (legend_x0, legend_y0),
        legend_x1 - legend_x0,
        legend_y1 - legend_y0,
        edgecolor='black',
        facecolor='none',
        lw=1,
        transform=ax.transAxes,
        clip_on=False
    )
)

plt.show()
