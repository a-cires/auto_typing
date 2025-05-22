import cv2
import pygame
import keyboard
import numpy as np

class TypingGUI:
    def __init__(self, width, height):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("TypingGUI")
        self.font = pygame.font.SysFont(None, 24)

    def plot(self, frame, translation=None, rotation=None):
        tx, ty, tz = translation
        yaw, pitch, roll = rotation

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame_surface = pygame.surfarray.make_surface(frame)
        self.screen.blit(frame_surface, (0, 0))

        self.draw_translation_arrows(tx, ty, tz)
        self.draw_rotation_arrows(pitch, yaw)
        self.draw_roll_arrow(roll)

        pygame.display.update()

    def plot_with_labels(self, frame, translation=None, rotation=None, raw_translation=None, raw_rotation=None):
        tx, ty, tz = translation
        yaw, pitch, roll = rotation

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame_surface = pygame.surfarray.make_surface(frame)
        self.screen.blit(frame_surface, (0, 0))

        self.draw_translation_arrows(tx, ty, tz)
        self.draw_rotation_arrows(pitch, yaw)
        self.draw_roll_arrow(roll)

        width, height = self.screen.get_size()

        # Use raw values if provided, else fallback to discretized
        r_tx, r_ty, r_tz = raw_translation if raw_translation else (tx, ty, tz)
        r_yaw, r_pitch, r_roll = raw_rotation if raw_rotation else (yaw, pitch, roll)

        self._draw_text(f"X: {r_tx:.2f}", width // 2, height - 140)
        self._draw_text(f"Y: {r_ty:.2f}", 20, height - 140)
        self._draw_text(f"Z: {r_tz:.2f}", 20, height - 100)
        self._draw_text(f"Pitch: {r_pitch:.2f}", width - 200, height - 140)
        self._draw_text(f"Yaw: {r_yaw:.2f}", width - 200, height - 100)
        self._draw_text(f"Roll: {r_roll:.2f}", width // 2 - 30, height // 2 - 90)

        pygame.display.update()

    def _draw_text(self, text, x, y):
        shadow = self.font.render(text, True, (0, 0, 0))
        self.screen.blit(shadow, (x + 1, y + 1))
        label = self.font.render(text, True, (255, 255, 255))
        self.screen.blit(label, (x, y))

    def plot_from_raw_motion(self, frame, tx, ty, tz, yaw, pitch, roll):
        d_tx, d_ty, d_tz, d_yaw, d_pitch, d_roll = self.discretize_motion(tx, ty, tz, yaw, pitch, roll)
        self.plot_with_labels(
            frame,
            translation=[d_tx, d_ty, d_tz],
            rotation=[d_yaw, d_pitch, d_roll],
            raw_translation=[tx, ty, tz],
            raw_rotation=[yaw, pitch, roll]
        )

    def axis_from_keys(self, pos_key, neg_key):
        pos = keyboard.is_pressed(pos_key)
        neg = keyboard.is_pressed(neg_key)
        return int(pos) - int(neg)

    def discretize_motion(self, tx, ty, tz, yaw, pitch, roll):
        def to_direction(val):
            return 1 if val > 0 else -1 if val < 0 else 0
        return (
            to_direction(tx),
            to_direction(ty),
            to_direction(tz),
            to_direction(yaw),
            to_direction(pitch),
            to_direction(roll)
        )

    def draw_translation_arrows(self, tx, ty, tz):
        width, height = self.screen.get_size()
        size = 20
        base_left_x, base_left_y = 60, height - 60
        base_center_x = width // 2
        base_center_y = height - 60

        self._draw_text("Translation (Y/Z)", 20, height - 120)
        self._draw_text("Translation (X)", base_center_x - 60, height - 120)

        arrows = {
            'left':  {'points': [(base_left_x - 2*size, base_left_y), (base_left_x - size, base_left_y - size), (base_left_x - size, base_left_y + size)], 'active': tz == 1},
            'right': {'points': [(base_left_x + 2*size, base_left_y), (base_left_x + size, base_left_y - size), (base_left_x + size, base_left_y + size)], 'active': tz == -1},
            'up':    {'points': [(base_left_x, base_left_y - 2*size), (base_left_x - size, base_left_y - size), (base_left_x + size, base_left_y - size)], 'active': ty == 1},
            'down':  {'points': [(base_left_x, base_left_y + 2*size), (base_left_x - size, base_left_y + size), (base_left_x + size, base_left_y + size)], 'active': ty == -1},
            'forward':  {'points': [(base_center_x, base_center_y - 2*size), (base_center_x - size, base_center_y - size), (base_center_x + size, base_center_y - size)], 'active': tx == 1},
            'backward': {'points': [(base_center_x, base_center_y + 2*size), (base_center_x - size, base_center_y + size), (base_center_x + size, base_center_y + size)], 'active': tx == -1}
        }

        for arrow in arrows.values():
            color = (144, 238, 144) if arrow['active'] else (255, 0, 0)
            alpha = 255 if arrow['active'] else 100
            arrow_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            pygame.draw.polygon(arrow_surface, (*color, alpha), arrow['points'])
            self.screen.blit(arrow_surface, (0, 0))

    def draw_rotation_arrows(self, pitch, yaw):
        width, height = self.screen.get_size()
        size = 20
        base_right_x, base_right_y = width - 60, height - 60

        self._draw_text("Rotation (Pitch/Yaw)", width - 200, height - 120)

        arrows = {
            'pitch_up':    {'points': [(base_right_x, base_right_y - 2*size), (base_right_x - size, base_right_y - size), (base_right_x + size, base_right_y - size)], 'active': pitch == 1},
            'pitch_down':  {'points': [(base_right_x, base_right_y + 2*size), (base_right_x - size, base_right_y + size), (base_right_x + size, base_right_y + size)], 'active': pitch == -1},
            'yaw_left':  {'points': [(base_right_x - 2*size, base_right_y), (base_right_x - size, base_right_y - size), (base_right_x - size, base_right_y + size)], 'active': yaw == 1},
            'yaw_right': {'points': [(base_right_x + 2*size, base_right_y), (base_right_x + size, base_right_y - size), (base_right_x + size, base_right_y + size)], 'active': yaw == -1}
        }

        for arrow in arrows.values():
            color = (144, 238, 144) if arrow['active'] else (255, 0, 0)
            alpha = 255 if arrow['active'] else 100
            arrow_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            pygame.draw.polygon(arrow_surface, (*color, alpha), arrow['points'])
            self.screen.blit(arrow_surface, (0, 0))

    def draw_roll_arrow(self, roll):
        width, height = self.screen.get_size()
        center_x, center_y = width // 2, height // 2
        radius = 60
        arc_surface = pygame.Surface((width, height), pygame.SRCALPHA)

        if roll == 1:
            start_angle = np.pi * 0.2
            end_angle = np.pi
            arc_center = (center_x, center_y - 40)
            head_x = arc_center[0] - radius + 3
        elif roll == -1:
            start_angle = 0
            end_angle = np.pi * 0.8
            arc_center = (center_x, center_y - 40)
            head_x = arc_center[0] + radius - 4
        else:
            start_angle = 0
            end_angle = np.pi * 0.8
            arc_center = (center_x, center_y - 40)
            head_x = arc_center[0] + radius - 4

        head_y = arc_center[1] + 5
        is_active = roll != 0
        color = (144, 238, 144) if is_active else (255, 0, 0)
        alpha = 255 if is_active else 100

        arc_rect = pygame.Rect(
            arc_center[0] - radius,
            arc_center[1] - radius,
            2 * radius,
            2 * radius
        )

        pygame.draw.arc(arc_surface, (*color, alpha), arc_rect, start_angle, end_angle, 6)

        arrow_tip = (int(head_x), int(head_y))
        left = (arrow_tip[0] - 10, arrow_tip[1] - 5)
        right = (arrow_tip[0] + 10, arrow_tip[1] - 5)

        pygame.draw.polygon(arc_surface, (*color, alpha), [arrow_tip, left, right])
        self.screen.blit(arc_surface, (0, 0))


def main_manual():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    gui = TypingGUI(width, height)

    raw_tx = 0.8
    raw_ty = -0.3
    raw_tz = 0
    raw_yaw = 0.5
    raw_pitch = -1.2
    raw_roll = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        gui.plot_from_raw_motion(frame, raw_tx, raw_ty, raw_tz, raw_yaw, raw_pitch, raw_roll)

        if keyboard.is_pressed('p'):
            break

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main_manual()
