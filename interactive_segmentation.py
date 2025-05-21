import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # Define project root early
# Ensure the project root is in sys.path to prioritize the local 'sam2' package
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

# --- Configuration ---
video_dir = os.path.join(_PROJECT_ROOT, "./videos/sfu-video-short")  # Absolute path
sam2_checkpoint = os.path.join(_PROJECT_ROOT, "checkpoints/sam2.1_hiera_large.pt")

# For Hydra, config_name is resolved relative to search paths like pkg://sam2 (the sam2 package dir).
# The config file exists at /Users/seboes/Code/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml
# Since pkg://sam2 points to /Users/seboes/Code/sam2/sam2/, the config name should be configs/sam2.1/sam2.1_hiera_l
model_cfg = "configs/sam2.1/sam2.1_hiera_l"  # No need for "../" since it's in the sam2 package dir
# ---

# --- Device Setup ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
# ---


# --- Helper Functions (from original script) ---
def show_mask(mask, ax, obj_id=None, random_color=False):
    """Show segmentation mask similar to the original sam-2-cockpit-segmentation.py."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id % 10
        color = np.array([*cmap(cmap_idx)[:3], 0.6])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    # This is the key change - we're using a separate call to imshow specifically for the mask
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    """Display points for segmentation interactions."""
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


# ---


# --- Main Application ---
class InteractiveSegmenter:
    def __init__(self, predictor_instance, image_path_val):
        self.predictor = predictor_instance
        self.image_path = image_path_val
        self.image = Image.open(self.image_path)
        self.points_by_id = {}  # Store points for each object ID
        self.labels_by_id = {}  # Store labels for each object ID
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.obj_id_counter = 1
        self.current_obj_id = self.obj_id_counter
        # Store all masks for all objects
        self.masks_by_id = {}
        # Store video information
        self.video_dir = os.path.dirname(self.image_path)
        self.frame_names = sorted(
            [
                p
                for p in os.listdir(self.video_dir)
                if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
            ],
            key=lambda p: int(os.path.splitext(p)[0]),
        )

        # Initialize SAM2 state for a single image (even though it's a video predictor)
        # We'll treat the single image as a one-frame video.
        # Create a temporary directory for the single frame if it doesn't exist
        self.temp_video_dir = "./temp_single_frame_video"
        os.makedirs(self.temp_video_dir, exist_ok=True)
        temp_image_path = os.path.join(self.temp_video_dir, "0.jpg")
        self.image.save(temp_image_path)

        self.inference_state = self.predictor.init_state(video_path=self.temp_video_dir)

        self.update_plot()
        print(
            "Click on the image to add points. Left-click for positive, Right-click for negative."
        )
        print("Press 'n' to segment the current object and start a new one.")
        print("Press 'r' to reset all points for the current object.")
        print("Press 'c' to clear all segmentations and start over.")
        print("Press 'a' to display all segmentation masks with their IDs.")
        print(
            "Press 'p' to propagate segmentation across all video frames and save results."
        )
        print("Press 's' to save the current view.")
        print(
            "Press 'd' to save a clean version of the current view (without matplotlib elements)."
        )
        print("Close the plot window to exit.")
        self.fig.canvas.mpl_connect("key_press_event", self.onkey)

    def onclick(self, event):
        if event.inaxes != self.ax:
            return

        x, y = int(event.xdata), int(event.ydata)

        # Initialize points and labels for this object if not exists
        if self.current_obj_id not in self.points_by_id:
            self.points_by_id[self.current_obj_id] = []
            self.labels_by_id[self.current_obj_id] = []

        # Left click: positive point (label 1)
        # Right click: negative point (label 0)
        # Middle click: ignored for now
        if event.button == 1:
            self.points_by_id[self.current_obj_id].append([x, y])
            self.labels_by_id[self.current_obj_id].append(1)
            print(f"Added positive point: ({x}, {y})")
        elif event.button == 3:
            self.points_by_id[self.current_obj_id].append([x, y])
            self.labels_by_id[self.current_obj_id].append(0)
            print(f"Added negative point: ({x}, {y})")
        else:
            return  # Ignore other clicks

        self.predict_and_update()

    def onkey(self, event):
        if event.key == "r":  # Reset points for current object
            print("Resetting points for current object.")
            if self.current_obj_id in self.points_by_id:
                self.points_by_id[self.current_obj_id] = []
                self.labels_by_id[self.current_obj_id] = []
            # Remove this object from masks dictionary
            if self.current_obj_id in self.masks_by_id:
                del self.masks_by_id[self.current_obj_id]
            self.update_plot()
        elif event.key == "n":  # New object
            self.obj_id_counter += 1
            self.current_obj_id = self.obj_id_counter
            print(f"Starting new object (ID: {self.current_obj_id}).")
            # No need to clear masks here, they are re-drawn on update_plot
            self.update_plot(clear_previous_masks=False)  # Keep old masks
        elif event.key == "c":  # Clear all
            print("Clearing all segmentations and points.")
            self.points_by_id = {}
            self.labels_by_id = {}
            self.obj_id_counter = 1
            self.current_obj_id = self.obj_id_counter
            # Clear all stored masks
            self.masks_by_id = {}
            # Reset inference state to clear all previous segmentations from the model
            self.inference_state = self.predictor.init_state(
                video_path=self.temp_video_dir
            )
            self.update_plot()
        elif event.key == "a":  # Display all segmentation masks with their IDs
            print(
                f"Displaying all segmentation masks. Total objects: {len(self.masks_by_id)}"
            )
            self.display_all_masks()
        elif (
            event.key == "p"
        ):  # Propagate segmentation across all video frames and save results
            self.propagate_segmentation()
        elif event.key == "s":  # Save current view
            output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "segmentation_output"
            )
            os.makedirs(output_dir, exist_ok=True)

            # Save the figure with the current view
            output_path = os.path.join(
                output_dir,
                f"segmentation_{len(self.points_by_id[self.current_obj_id])}_points.png",
            )
            self.fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved current view to {output_path}")

            # If we have a mask, also save it separately
            if hasattr(self, "last_mask") and self.last_mask is not None:
                mask_path = os.path.join(
                    output_dir,
                    f"mask_{len(self.points_by_id[self.current_obj_id])}_points.png",
                )
                plt.figure(figsize=(8, 8))
                plt.imshow(self.last_mask, cmap="gray")
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(mask_path)
                plt.close()
                print(f"Saved mask to {mask_path}")
        elif (
            event.key == "d"
        ):  # Save a clean version of the current view (without matplotlib elements)
            output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "segmentation_output"
            )
            os.makedirs(output_dir, exist_ok=True)

            # Save the clean image
            clean_output_path = os.path.join(
                output_dir,
                f"clean_view_{len(self.points_by_id[self.current_obj_id])}_points.png",
            )
            self.save_clean_view(clean_output_path)

    def predict_and_update(self):
        """Predict segmentation based on current points and update the display."""
        if (
            self.current_obj_id not in self.points_by_id
            or not self.points_by_id[self.current_obj_id]
        ):
            self.update_plot()
            return

        points_np = np.array(self.points_by_id[self.current_obj_id], dtype=np.float32)
        labels_np = np.array(self.labels_by_id[self.current_obj_id], dtype=np.int32)

        # We are always interacting with the first frame (frame_idx=0)
        # The obj_id helps distinguish different objects if you segment multiple.
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=0,  # Always frame 0 for a single image
            obj_id=self.current_obj_id,
            points=points_np,
            labels=labels_np,
        )

        # Store masks for later display
        if len(out_mask_logits) > 0 and len(out_obj_ids) > 0:
            for i, obj_id in enumerate(out_obj_ids):
                if i < len(out_mask_logits) and out_mask_logits[i] is not None:
                    # Convert mask logits to binary mask
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy()

                    # If no pixels in mask, try a lower threshold
                    if np.count_nonzero(mask) == 0:
                        mask = (out_mask_logits[i] > -0.3).cpu().numpy()

                    # Store the mask for this object ID
                    self.masks_by_id[obj_id] = mask

        self.update_plot(out_mask_logits, out_obj_ids)

    def update_plot(self, mask_logits=None, obj_ids=None, clear_previous_masks=True):
        """Update the plot with the current image, masks, and points."""
        # Clear the current axes
        self.ax.clear()

        # First show the background image
        self.ax.imshow(self.image)
        self.ax.set_title(
            f"Click to add points. Current Object ID: {self.current_obj_id}"
        )

        # Then show masks if available
        if mask_logits is not None and obj_ids is not None:
            for i, obj_id in enumerate(obj_ids):
                # Only show masks for the current object or all objects if not clearing previous ones
                if obj_id == self.current_obj_id or not clear_previous_masks:
                    if i < len(mask_logits) and mask_logits[i] is not None:
                        # Convert mask logits to binary mask using threshold
                        mask = (mask_logits[i] > 0.0).cpu().numpy()

                        # If no pixels in mask, try a lower threshold
                        if np.count_nonzero(mask) == 0:
                            mask = (mask_logits[i] > -0.3).cpu().numpy()

                        # Show the mask
                        show_mask(mask, self.ax, obj_id=obj_id)

                        # Store for potential future reference
                        self.last_mask = mask

        # Finally show the clicked points on top of everything
        if (
            self.current_obj_id in self.points_by_id
            and self.points_by_id[self.current_obj_id]
        ):
            show_points(
                np.array(self.points_by_id[self.current_obj_id]),
                np.array(self.labels_by_id[self.current_obj_id]),
                self.ax,
            )

        # Force the figure to update
        self.fig.canvas.draw()

    def display_all_masks(self):
        """Display all segmentation masks with their object IDs clearly labeled."""
        # Clear the current axes
        self.ax.clear()

        # First show the background image
        self.ax.imshow(self.image)
        self.ax.set_title("All Segmentation Masks with IDs")

        # Check if we have any masks
        if not self.masks_by_id:
            self.ax.text(
                0.5,
                0.5,
                "No masks available. Add points first.",
                ha="center",
                va="center",
                fontsize=12,
                transform=self.ax.transAxes,
            )
            self.fig.canvas.draw()
            return

        # Show all masks
        for obj_id, mask in self.masks_by_id.items():
            # Display the mask
            show_mask(mask, self.ax, obj_id=obj_id)

            # Find the center of mass of the mask to place a label
            if np.any(mask):
                # Make sure mask is 2D before finding indices
                if mask.ndim > 2:
                    # If mask has more than 2 dimensions, flatten extra dimensions
                    mask_2d = mask.reshape(mask.shape[-2], mask.shape[-1])
                else:
                    mask_2d = mask

                # Get coordinates of mask pixels
                indices = np.where(mask_2d)
                if len(indices) == 2 and len(indices[0]) > 0:
                    y_indices, x_indices = indices

                    # Calculate center of mass
                    center_y = int(np.mean(y_indices))
                    center_x = int(np.mean(x_indices))

                    # Add a text label with the object ID
                    self.ax.text(
                        center_x,
                        center_y,
                        f"ID: {obj_id}",
                        color="white",
                        fontsize=12,
                        fontweight="bold",
                        ha="center",
                        va="center",
                        bbox=dict(boxstyle="round", fc="black", alpha=0.7),
                    )

        # Add a legend showing color - object ID mapping
        cmap = plt.get_cmap("tab10")
        legend_elements = []
        for obj_id in sorted(self.masks_by_id.keys()):
            color = cmap(obj_id % 10)[:3]
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    label=f"Object {obj_id}",
                    markerfacecolor=color,
                    markersize=10,
                )
            )

        if legend_elements:
            self.ax.legend(handles=legend_elements, loc="upper right")

        # Force the figure to update
        self.fig.canvas.draw()

    def propagate_segmentation(self):
        """Propagate segmentation across all video frames and save results."""
        # Check if we have any segmentation to propagate
        if not self.masks_by_id:
            print("No segmentation masks available. Add points first.")
            return

        # Create output directory
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

        # Get total number of frames
        total_frames = len(self.frame_names)
        print(f"Processing {total_frames} frames...")

        # Process in batches to manage memory
        batch_size = 300  # Process 100 frames at a time
        num_batches = (total_frames + batch_size - 1) // batch_size

        # Get the points and labels for each object
        for obj_id in self.masks_by_id.keys():
            if obj_id not in self.points_by_id or not self.points_by_id[obj_id]:
                print(
                    f"No points available for object {obj_id}. Please add points first."
                )
                return

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_frames)
            print(
                f"\nProcessing batch {batch_idx + 1}/{num_batches} (frames {start_idx}-{end_idx - 1})"
            )

            # Create a temporary directory for this batch
            temp_dir = os.path.join(self.temp_video_dir, f"batch_{batch_idx}")
            os.makedirs(temp_dir, exist_ok=True)

            # Copy selected frames to temp directory with sequential numbering
            print("Preparing frames for this batch...")
            for i, frame_idx in enumerate(range(start_idx, end_idx)):
                src_path = os.path.join(self.video_dir, self.frame_names[frame_idx])
                dst_path = os.path.join(temp_dir, f"{i:05d}.jpg")
                img = Image.open(src_path)
                img.save(dst_path)

            try:
                # Initialize inference state for this batch
                print("Initializing inference state for batch...")
                inference_state = self.predictor.init_state(video_path=temp_dir)

                # First add our current segmentation to the inference state
                print(f"Adding initial segmentation to batch...")
                # Apply to each object ID
                for obj_id in self.masks_by_id.keys():
                    initial_points = np.array(
                        self.points_by_id[obj_id], dtype=np.float32
                    )
                    initial_labels = np.array(self.labels_by_id[obj_id], dtype=np.int32)
                    self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,  # First frame
                        obj_id=obj_id,
                        points=initial_points,
                        labels=initial_labels,
                    )

                # Run propagation and collect the results in a dict
                video_segments = {}  # video_segments contains the per-frame segmentation results

                # Now propagate through the frames in this batch
                frame_count = 0
                for (
                    out_frame_idx,
                    out_obj_ids,
                    out_mask_logits,
                ) in self.predictor.propagate_in_video(inference_state):
                    # Store segmentations for this frame
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                # Process and save each frame with masks
                print("Saving frames with masks...")

                # Create a directory for masks if it doesn't exist
                masks_dir = os.path.join(output_dir, "masks")
                os.makedirs(masks_dir, exist_ok=True)

                for i, frame_idx in enumerate(sorted(video_segments.keys())):
                    # Get the original frame index
                    original_frame_idx = start_idx + frame_idx

                    # Save masks for this frame
                    frame_masks = video_segments[frame_idx]
                    masks_output_path = os.path.join(
                        masks_dir, f"frame_{original_frame_idx:04d}_masks.npy"
                    )
                    np.save(masks_output_path, frame_masks)

                    # Load the frame image (from original directory)
                    frame_path = os.path.join(
                        self.video_dir, self.frame_names[original_frame_idx]
                    )
                    frame_image = Image.open(frame_path)

                    # Save clean version of the image with mask overlay
                    clean_result = self.create_clean_segmentation_overlay(
                        frame_image, video_segments[frame_idx]
                    )

                    # Save the clean image
                    clean_output_path = os.path.join(
                        output_dir, f"frame_{original_frame_idx:04d}.png"
                    )
                    clean_result.save(clean_output_path)

                    # Create a figure for this frame (for reference/debugging)
                    fig, ax = plt.subplots(figsize=(10, 7))
                    ax.imshow(frame_image)
                    ax.set_title(f"Frame {original_frame_idx}")

                    # Overlay all masks for this frame
                    for obj_id, mask in video_segments[frame_idx].items():
                        show_mask(mask, ax, obj_id=obj_id)

                        # Find center of mask for label (if mask is not empty)
                        if np.any(mask):
                            # Make sure mask is 2D
                            if mask.ndim > 2:
                                mask_2d = mask.reshape(mask.shape[-2], mask.shape[-1])
                            else:
                                mask_2d = mask

                            indices = np.where(mask_2d)
                            if len(indices) == 2 and len(indices[0]) > 0:
                                y_indices, x_indices = indices
                                center_y = int(np.mean(y_indices))
                                center_x = int(np.mean(x_indices))

                                # Add object ID label
                                ax.text(
                                    center_x,
                                    center_y,
                                    f"ID: {obj_id}",
                                    color="white",
                                    fontsize=12,
                                    fontweight="bold",
                                    ha="center",
                                    va="center",
                                    bbox=dict(boxstyle="round", fc="black", alpha=0.7),
                                )

                    # Add a legend
                    cmap = plt.get_cmap("tab10")
                    legend_elements = []
                    for obj_id in sorted(video_segments[frame_idx].keys()):
                        color = cmap(obj_id % 10)[:3]
                        legend_elements.append(
                            plt.Line2D(
                                [0],
                                [0],
                                marker="s",
                                color="w",
                                label=f"Object {obj_id}",
                                markerfacecolor=color,
                                markersize=10,
                            )
                        )
                    if legend_elements:
                        ax.legend(handles=legend_elements, loc="upper right")

                    # Save the annotated figure
                    viz_output_dir = os.path.join(output_dir, "annotated")
                    os.makedirs(viz_output_dir, exist_ok=True)
                    viz_output_path = os.path.join(
                        viz_output_dir, f"frame_{original_frame_idx:04d}_annotated.png"
                    )
                    fig.savefig(viz_output_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)

                    # Print progress every 10 frames
                    if i % 10 == 0:
                        print(
                            f"Saved frame {i + 1}/{len(video_segments)} (original frame {original_frame_idx})"
                        )

                # Clean up temporary files for this batch
                try:
                    import shutil

                    shutil.rmtree(temp_dir)
                except PermissionError:
                    print(
                        f"Warning: Could not remove temporary directory {temp_dir}. You may need to remove it manually."
                    )

                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                plt.close("all")  # Close any open figures
                if "Invalid buffer size" in str(e):
                    print("\nERROR: Memory limit exceeded when processing batch.")
                    print(
                        "Try reducing the batch size or using a machine with more memory."
                    )
                else:
                    print(f"\nError during batch processing: {e}")
                return

        print(f"Segmentation propagation complete! All frames saved to {output_dir}")

        # Display a random sample of the saved frames
        self.show_propagation_results(output_dir)

    def show_propagation_results(self, output_dir):
        """Display a preview of propagation results by showing a sample of frames."""
        # Get a list of all saved frames
        saved_frames = sorted(
            [f for f in os.listdir(output_dir) if f.endswith("_segmented.png")]
        )

        if not saved_frames:
            print("No frames found in output directory.")
            return

        # Select frames to display (first, last, and few in between)
        num_frames = len(saved_frames)
        if num_frames <= 6:
            sample_frames = saved_frames
        else:
            # Take first, last, and 4 evenly spaced frames in between
            indices = [0]
            indices.extend([int(num_frames * i / 5) for i in range(1, 5)])
            indices.append(num_frames - 1)
            sample_frames = [saved_frames[i] for i in indices]

        # Create a figure to display the sample
        num_cols = min(3, len(sample_frames))
        num_rows = (len(sample_frames) + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        if num_rows * num_cols == 1:
            axes = np.array([axes])  # Make iterable if only one subplot
        axes = axes.flatten()

        # Display each sample frame
        for i, frame_file in enumerate(sample_frames):
            if i < len(axes):
                # Load and display the image
                img = plt.imread(os.path.join(output_dir, frame_file))
                axes[i].imshow(img)

                # Extract frame number from filename
                frame_num = int(frame_file.split("_")[1])
                axes[i].set_title(f"Frame {frame_num}")
                axes[i].axis("off")

        # Hide empty subplots
        for i in range(len(sample_frames), len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.suptitle("Propagation Results Preview (Sample Frames)", fontsize=16, y=1.02)
        plt.show()

    def create_clean_segmentation_overlay(self, image, masks_dict):
        """Create a clean image with segmentation overlay without matplotlib elements.

        Args:
            image: PIL Image object of the original frame
            masks_dict: Dictionary mapping object IDs to masks

        Returns:
            PIL Image with the segmentation overlay
        """
        # Convert image to numpy array if it's not already
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        # Create a copy of the image to overlay masks
        result = img_array.copy()

        # Get colormap
        cmap = plt.get_cmap("tab10")

        # Overlay each mask with its color
        for obj_id, mask in masks_dict.items():
            # Make sure mask is properly shaped
            if mask.ndim > 2:
                mask_2d = mask.reshape(mask.shape[-2], mask.shape[-1])
            else:
                mask_2d = mask

            # Skip if mask is empty
            if not np.any(mask_2d):
                continue

            # Get color for this object ID
            color_idx = obj_id % 10
            color_rgb = np.array(cmap(color_idx)[:3]) * 255  # Convert to 0-255 range

            # Create colored mask
            colored_mask = np.zeros(
                (mask_2d.shape[0], mask_2d.shape[1], 3), dtype=np.uint8
            )
            for i, color in enumerate(color_rgb):
                colored_mask[:, :, i] = mask_2d * color

            # Create alpha channel for the mask (50% opacity)
            alpha = mask_2d * 0.5

            # Apply the colored mask to the image
            for c in range(3):  # RGB channels
                result[:, :, c] = (
                    result[:, :, c] * (1 - alpha) + colored_mask[:, :, c] * alpha
                ).astype(np.uint8)

        # Convert back to PIL Image
        return Image.fromarray(result)

    def save_clean_view(self, output_path):
        """Save a clean version of the current view without matplotlib elements."""
        if not self.masks_by_id:
            # If no masks, just save the original image
            print("No masks available. Saving original image.")
            self.image.save(output_path)
            print(f"Saved clean view to {output_path}")
            return

        # Create clean image with segmentation overlays
        clean_result = self.create_clean_segmentation_overlay(
            self.image, self.masks_by_id
        )

        # Save the clean image
        clean_result.save(output_path)
        print(f"Saved clean view to {output_path}")

    def run(self):
        plt.show()
        # Clean up the temporary directory
        if os.path.exists(self.temp_video_dir):
            try:
                for f_name in os.listdir(self.temp_video_dir):
                    try:
                        os.remove(os.path.join(self.temp_video_dir, f_name))
                    except PermissionError:
                        print(
                            f"Warning: Could not remove {f_name}. You may need to remove it manually."
                        )
                os.rmdir(self.temp_video_dir)
            except PermissionError:
                print(
                    f"Warning: Could not remove temporary directory {self.temp_video_dir}. You may need to remove it manually."
                )


if __name__ == "__main__":
    # --- Load SAM2 Model ---
    print("Loading SAM2 model...")
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    print("Model loaded.")
    # ---

    # --- Setup Video Frames ---
    if not os.path.exists(video_dir):
        print(f"Error: Video directory '{video_dir}' not found.")
        print("Please create it and place JPEG frames (e.g., 0.jpg, 1.jpg) inside.")
        exit()

    frame_names = [
        p
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    if not frame_names:
        print(f"Error: No JPEG frames found in '{video_dir}'.")
        exit()

    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    first_frame_path = os.path.join(video_dir, frame_names[0])
    # ---

    print(f"Loading first frame: {first_frame_path}")

    app = InteractiveSegmenter(predictor, first_frame_path)
    app.run()
