import numpy as np
import torch
from typing import List, Tuple
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from collections import deque
from cotracker.predictor import CoTrackerPredictor
import os

class TraceProcessor:
    def __init__(self, cotracker_model_path, begin_track_step: int = 10, redraw_frequency: int = 25, 
                 num_points: int = 5, buffer_size: int = 10, window_size: int = 15, device: int = 0):
        self.cotracker_model = CoTrackerPredictor(
            checkpoint=os.path.join(
                cotracker_model_path
            )
        ).to(device)
        self.begin_track_step = begin_track_step
        self.redraw_frequency = redraw_frequency
        self.num_points = num_points
        self.buffer_size = buffer_size
        self.window_size = window_size
        self.device = device
        
        self.step = 0
        self.traced = False
        self.trace_buffer = None
        self.image_buffer = deque(maxlen=self.buffer_size)
        self.last_trace_step = 0
        self.image_overlaid_list: List[Image.Image] = []

    def process_image(self, image: Image.Image) -> Image.Image:
        self.step += 1
        image_array = np.array(image, dtype=np.uint8).transpose(2, 0, 1)[None, :]
        self.image_buffer.append(image_array)
        
        if self.step < self.begin_track_step:
            self.traced = False
            image_overlaid = image
            has_trace = False
        else:
            self._update_trace()
            if self.traced:
                has_trace = True
                trace = self.trace_buffer[-self.window_size:]
                image_overlaid = self._visualize_trace(image.copy(), trace)
            else:
                has_trace = False
                image_overlaid = image
        
        self.image_overlaid_list.append(image_overlaid)
        return image_overlaid, has_trace

    def _update_trace(self):
        if (self.step % self.redraw_frequency == 0 or 
            self.step == self.begin_track_step or 
            (not self.traced and self.step - self.last_trace_step > 5)):
            self._run_cotracker()
        elif self.trace_buffer is not None:
            self._update_trace_buffer()

    def _run_cotracker(self):
        video = torch.from_numpy(np.concatenate(list(self.image_buffer), axis=0)).to(device=self.device)
        video = video.unsqueeze(0).float()
        with torch.no_grad():
            pred_tracks, pred_visibility = self.cotracker_model(video, grid_size=30)
        trace = self._filter_points(pred_tracks[0].cpu().numpy(), (336, 336))
        distance = np.mean(np.sum(np.abs(trace[1:] - trace[:-1]), axis=-1), axis=0)
        ids = np.where(distance > 1.0)[0]
        if ids.shape[0] > self.num_points:
            sampled_ids = np.random.choice(ids, size=self.num_points, replace=False)
            self.traced = True
            self.trace_buffer = trace[:, sampled_ids]
        else:
            print(f"Step {self.step} cannot be traced by cotracker, number of active points:{ids.shape[0]}")
            self.traced = False
            self.last_trace_step = self.step

    def _update_trace_buffer(self):
        points_coord = self.trace_buffer[-1]
        queries = np.concatenate([np.ones((self.num_points, 1)) * (self.buffer_size - 2), points_coord], axis=1)[None, :]
        video = torch.from_numpy(np.concatenate(list(self.image_buffer), axis=0)).to(device=self.device)
        video = video.unsqueeze(0).float()
        with torch.no_grad():
            pred_tracks, pred_visibility = self.cotracker_model(video, queries=torch.from_numpy(queries).float().to(device=self.device))
        self.trace_buffer = np.concatenate([self.trace_buffer, pred_tracks[0, -1:].cpu().numpy()], axis=0)

    @staticmethod
    def _filter_points(trace: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
        height, width = img_shape
        mask = (trace[..., 0] >= 0) & (trace[..., 0] < width) & (trace[..., 1] >= 0) & (trace[..., 1] < height)
        valid_points_mask = np.all(mask, axis=0)
        filtered_trace = trace[:, valid_points_mask, :]
        return filtered_trace

    @staticmethod
    def _visualize_trace(image: Image.Image, trace: np.ndarray, linewidth: int = 2, arrow_length: int = 10, arrow_angle: int = 40) -> Image.Image:
        draw = ImageDraw.Draw(image)
        num_steps_to_trace, num_points, _ = trace.shape
        colors = plt.cm.get_cmap('hsv', num_points+1)
        if num_points > 1:
            for point_idx in range(num_points):
                color = tuple((np.array(colors(point_idx)[:3]) * 255).astype(int))
                for step in range(num_steps_to_trace - 1):
                    start_point = tuple(trace[step, point_idx])
                    end_point = tuple(trace[step + 1, point_idx])
                    draw.line([start_point, end_point], fill=color, width=linewidth)
        for point_idx in range(num_points):
            final_point = np.array(trace[-1, point_idx])
            prev_point = np.array(trace[-2, point_idx])
            color = tuple((np.array(colors(point_idx)[:3]) * 255).astype(int))
            direction = final_point - prev_point
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm != 0 else direction
            left_wing = np.array([
                final_point[0] - arrow_length * np.cos(np.deg2rad(arrow_angle)) * direction[0] + arrow_length * np.sin(np.deg2rad(arrow_angle)) * direction[1],
                final_point[1] - arrow_length * np.cos(np.deg2rad(arrow_angle)) * direction[1] - arrow_length * np.sin(np.deg2rad(arrow_angle)) * direction[0]
            ])
            right_wing = np.array([
                final_point[0] - arrow_length * np.cos(np.deg2rad(arrow_angle)) * direction[0] - arrow_length * np.sin(np.deg2rad(arrow_angle)) * direction[1],
                final_point[1] - arrow_length * np.cos(np.deg2rad(arrow_angle)) * direction[1] + arrow_length * np.sin(np.deg2rad(arrow_angle)) * direction[0]
            ])
            draw.polygon([tuple(final_point), tuple(left_wing), tuple(right_wing)], fill=color, outline=color)
        return image

    def get_processed_images(self) -> List[Image.Image]:
        return self.image_overlaid_list

    def reset(self):
        self.step = 0
        self.traced = False
        self.trace_buffer = None
        self.image_buffer.clear()
        self.last_trace_step = 0
        self.image_overlaid_list.clear()
