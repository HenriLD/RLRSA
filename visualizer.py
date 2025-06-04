import pygame
import sys
import os
import csv
from datetime import datetime
try:
    from environment import CooperativeChickenEnv
    from random_agents import run_random_agents_episode # Expects modified version
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# --- Constants ---
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
SIDE_PANEL_WIDTH = 300 # For dashboard metrics / replay info
GRID_AREA_WIDTH = SCREEN_WIDTH - SIDE_PANEL_WIDTH
GRID_AREA_HEIGHT = SCREEN_HEIGHT - 100 # Space for timeline/controls at bottom

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
LIGHT_GREY = (230, 230, 230)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
DARK_GREEN = (0, 100, 0)
BLUE = (0, 0, 200)
AGENT1_COLOR = (0, 150, 255) # Blueish
AGENT2_COLOR = (255, 150, 0) # Orangeish
CHICKEN_COLOR = (255, 255, 0) # Yellow
WALL_COLOR = (50, 50, 50) # Dark Grey
BUTTON_COLOR = (100, 150, 200)
BUTTON_HOVER_COLOR = (130, 180, 230)
TEXT_COLOR = BLACK
STATUS_SUCCESS_COLOR = (0, 128, 0) # Darker Green for success messages
STATUS_ERROR_COLOR = RED

# Fonts
pygame.init() # Initialize Pygame early for font loading
try:
    FONT_SMALL = pygame.font.Font(None, 24)
    FONT_MEDIUM = pygame.font.Font(None, 30)
    FONT_LARGE = pygame.font.Font(None, 36)
except Exception as e:
    print(f"Error loading default font: {e}. Using system font.")
    FONT_SMALL = pygame.font.SysFont(pygame.font.get_default_font(), 24)
    FONT_MEDIUM = pygame.font.SysFont(pygame.font.get_default_font(), 30)
    FONT_LARGE = pygame.font.SysFont(pygame.font.get_default_font(), 36)


# --- Predefined Grids ---
# Updated walls for new grid sizes
walls_small_10x10 = set([
    (2,2), (2,3), (2,4), (3,2), (4,2),  # Top-left L-shape
    (6,6), (6,7), (7,6),                # Mid-right small block
    (1,8), (2,8),                       # Top-right short vertical
    (8,1), (8,2),                       # Bottom-left short horizontal
    (5,5)                               # Central point
])

walls_medium_25x25 = set()
# Outer "frame" sections with gaps
for i in range(3, 22): # L/H = 25
    if not (10 <= i <= 14): # Create a wide central gap
        walls_medium_25x25.add((3, i))   # Top
        walls_medium_25x25.add((21, i))  # Bottom
        walls_medium_25x25.add((i, 3))   # Left
        walls_medium_25x25.add((i, 21))  # Right
# Some internal lines/blocks creating a cross-like pattern
walls_medium_25x25.update([(x, 12) for x in range(7, 18) if x != 12]) # Vertical line with gap at center
walls_medium_25x25.update([(12, y) for y in range(7, 18) if y != 12]) # Horizontal line with gap at center
walls_medium_25x25.update([(7,7), (7,8), (8,7), (8,8)])      # Small top-leftish block
walls_medium_25x25.update([(16,16), (16,17), (17,16), (17,17)])# Small bottom-rightish block
walls_medium_25x25.update([(7,16), (7,17), (8,16), (8,17)])      # Small top-rightish block
walls_medium_25x25.update([(16,7), (16,8), (17,7), (17,8)])# Small bottom-leftish block

walls_large_60x60 = set([(3,x) for x in range(30,55)])  # User's long horizontal line (top-mid-right)
walls_large_60x60.update([(7,x) for x in range(2,9)])   # User's short horizontal line (top-left)
walls_large_60x60.update([(x,2) for x in range(3,7)])   # User's short vertical line (top-left)
walls_large_60x60.update([(x,7) for x in range(3,7)])   # User's another short vertical (top-left)
walls_large_60x60.update([(4,5), (5,5)])                # User's two points (top-left)
# Add longer barriers / "zone" dividers
walls_large_60x60.update([(x, 29) for x in range(10, 50) if not (27 <= x <= 32)]) # Long vertical barrier near mid-left with a gap
walls_large_60x60.update([(29, y) for y in range(10, 50) if not (27 <= y <= 32)]) # Long horizontal barrier near mid-top with a gap
# Add some "rooms" or larger blocks (hollow squares with one opening)
# Room 1 (top-right quadrant)
for r_idx in range(10, 16): walls_large_60x60.add((r_idx, 40)); walls_large_60x60.add((r_idx, 45))
for c_idx in range(40, 46): walls_large_60x60.add((10, c_idx)); walls_large_60x60.add((15, c_idx))
if (12, 40) in walls_large_60x60: walls_large_60x60.remove((12,40)) # Opening
# Room 2 (bottom-left quadrant)
for r_idx in range(40, 46): walls_large_60x60.add((r_idx, 10)); walls_large_60x60.add((r_idx, 15))
for c_idx in range(10, 16): walls_large_60x60.add((40, c_idx)); walls_large_60x60.add((45, c_idx))
if (40, 12) in walls_large_60x60: walls_large_60x60.remove((40,12)) # Opening
# A few smaller, scattered obstacles
walls_large_60x60.update([(20,20), (20,21), (21,20), (21,21)]) # Small 2x2 block
walls_large_60x60.update([(50,50), (50,51), (51,50), (51,51)]) # Another Small 2x2 block
walls_large_60x60.update([(15,5), (16,5), (17,5), (15,6)]) # Small L-shape
walls_large_60x60.update([(5, 45), (5,46), (5,47), (6,47)]) # Small L-shape (mirrored)
walls_large_60x60.update([(x,x) for x in range(50,55)]) # Diagonal segment

PREDEFINED_GRIDS = {
    "Small (10x10)": {
        "L": 10, "H": 10, "max_steps": 100, # Increased max_steps slightly
        "walls": walls_small_10x10
    },
    "Medium (25x25)": {
        "L": 25, "H": 25, "max_steps": 500,
        "walls": walls_medium_25x25
    },
    "Large (60x60)": {
        "L": 60, "H": 60, "max_steps": 2000, # Adjusted max_steps from original 5000
        "walls": walls_large_60x60
    }
}

# --- Agent Types ---
AGENT_TYPES = ["Random"] # Only "Random" for now

# --- UI Element Helper ---
class Button:
    def __init__(self, x, y, width, height, text, color=BUTTON_COLOR, hover_color=BUTTON_HOVER_COLOR, font=FONT_MEDIUM, text_color=WHITE, selected_color=DARK_GREEN):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.font = font
        self.text_color = text_color
        self.selected_color = selected_color
        self.is_hovered = False
        self.is_selected = False

    def draw(self, surface):
        current_color = self.color
        if self.is_selected:
            current_color = self.selected_color
        elif self.is_hovered:
            current_color = self.hover_color
        
        pygame.draw.rect(surface, current_color, self.rect, border_radius=5)
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def check_hover(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)

    def is_clicked(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos)

# --- Main Application Class ---
class VisualizerApp:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("RL Coop Chicken Visualizer")
        self.clock = pygame.time.Clock()

        self.current_screen = "selection"  # "selection", "running_experiments", "dashboard", "replay"
        self.selected_agent_type = AGENT_TYPES[0]
        self.selected_grid_key = list(PREDEFINED_GRIDS.keys())[0] 

        self.experiment_results = []
        self.dashboard_metrics = {}
        self.status_message = "" 

        self.replay_game_index = -1
        self.replay_step_index = 0
        self.max_replay_steps = 0

        self._init_ui_elements()

    def _init_ui_elements(self):
        self.buttons = {}
        # Selection Screen
        self.buttons["run_experiments"] = Button(SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT - 100, 300, 50, "Run 10 Experiments")
        
        y_offset = 100
        agent_title_surf = FONT_MEDIUM.render("1. Select Agent Type:", True, BLACK)
        self.agent_title_rect = agent_title_surf.get_rect(topleft=(50, y_offset))
        y_offset += 40
        for i, agent_type in enumerate(AGENT_TYPES):
            btn = Button(50, y_offset + i * 50, 250, 40, agent_type)
            self.buttons[f"select_agent_{agent_type}"] = btn
        
        y_offset = 100
        grid_title_surf = FONT_MEDIUM.render("2. Select Grid Configuration:", True, BLACK)
        self.grid_title_rect = grid_title_surf.get_rect(topleft=(350, y_offset))
        y_offset += 40
        for i, grid_name in enumerate(PREDEFINED_GRIDS.keys()):
            btn = Button(350, y_offset + i * 50, 250, 40, grid_name)
            self.buttons[f"select_grid_{grid_name}"] = btn


        # Dashboard Screen
        self.buttons["back_to_selection"] = Button(20, SCREEN_HEIGHT - 70, 250, 50, "New Experiment Setup")
        self.game_replay_buttons = [] 

        # Replay Screen
        self.buttons["back_to_dashboard"] = Button(SIDE_PANEL_WIDTH + 20, SCREEN_HEIGHT - 70, 220, 50, "Back to Dashboard")
        self.buttons["prev_step"] = Button(GRID_AREA_WIDTH // 2 - 110, SCREEN_HEIGHT - 50, 100, 30, "< Prev")
        self.buttons["next_step"] = Button(GRID_AREA_WIDTH // 2 + 10, SCREEN_HEIGHT - 50, 100, 30, "Next >")
        self.timeline_rect = pygame.Rect(50, SCREEN_HEIGHT - 85, GRID_AREA_WIDTH - 100, 15) # For click detection
        self.agent_title_surf = agent_title_surf # Store pre-rendered surface
        self.grid_title_surf = grid_title_surf   # Store pre-rendered surface


    def run(self):
        running = True
        while running:
            mouse_pos = pygame.mouse.get_pos()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.handle_event(event)

            for button in self.buttons.values():
                if isinstance(button, Button): button.check_hover(mouse_pos)
            for button in self.game_replay_buttons:
                 if isinstance(button, Button): button.check_hover(mouse_pos)
            
            self._update_button_selected_states()
            self.render() # update() is mostly event-driven, render handles drawing current state
            
            pygame.display.flip()
            self.clock.tick(30) 

        pygame.quit()

    def _update_button_selected_states(self):
        if self.current_screen == "selection":
            for agent_type in AGENT_TYPES:
                self.buttons[f"select_agent_{agent_type}"].is_selected = (self.selected_agent_type == agent_type)
            for grid_name in PREDEFINED_GRIDS.keys():
                 self.buttons[f"select_grid_{grid_name}"].is_selected = (self.selected_grid_key == grid_name)


    def handle_event(self, event):
        if self.current_screen == "selection":
            if self.buttons["run_experiments"].is_clicked(event):
                self.current_screen = "running_experiments"
                self.status_message = "Running 10 experiments... please wait."
                # Force a screen update for the message
                self.screen.fill(WHITE)
                status_text_surf = FONT_MEDIUM.render(self.status_message, True, BLACK)
                self.screen.blit(status_text_surf, status_text_surf.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2)))
                pygame.display.flip()
                pygame.time.set_timer(pygame.USEREVENT + 1, 50, True) # Short delay then run

            for agent_type in AGENT_TYPES:
                if self.buttons[f"select_agent_{agent_type}"].is_clicked(event):
                    self.selected_agent_type = agent_type
            for grid_name in PREDEFINED_GRIDS.keys():
                if self.buttons[f"select_grid_{grid_name}"].is_clicked(event):
                    self.selected_grid_key = grid_name
        
        elif self.current_screen == "dashboard":
            if self.buttons["back_to_selection"].is_clicked(event):
                self.current_screen = "selection"
            for i, btn in enumerate(self.game_replay_buttons):
                if btn.is_clicked(event):
                    self.replay_game_index = i
                    self.replay_step_index = 0
                    if self.experiment_results and 0 <= i < len(self.experiment_results):
                        self.max_replay_steps = len(self.experiment_results[i]["history"])
                    else:
                        self.max_replay_steps = 0
                    self.current_screen = "replay"
                    break
        
        elif self.current_screen == "replay":
            if self.buttons["back_to_dashboard"].is_clicked(event):
                self.current_screen = "dashboard"
            if self.buttons["prev_step"].is_clicked(event):
                self.replay_step_index = max(0, self.replay_step_index - 1)
            if self.buttons["next_step"].is_clicked(event):
                self.replay_step_index = min(self.max_replay_steps -1, self.replay_step_index + 1)
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.timeline_rect.collidepoint(event.pos):
                 if self.max_replay_steps > 1: # Avoid division by zero if only one step
                    progress = (event.pos[0] - self.timeline_rect.x) / self.timeline_rect.width
                    self.replay_step_index = int(progress * (self.max_replay_steps -1)) # -1 because index
                    self.replay_step_index = max(0, min(self.max_replay_steps -1, self.replay_step_index))

        if event.type == pygame.USEREVENT + 1 and self.current_screen == "running_experiments":
            pygame.time.set_timer(pygame.USEREVENT + 1, 0) # Clear timer
            self._run_experiments_logic()
    
    def _export_metrics_to_csv(self):
        if not self.dashboard_metrics or "error" in self.dashboard_metrics or not self.experiment_results:
            self.status_message = "No metrics to export."; print("Export CSV: No metrics to export.")
            pygame.time.set_timer(pygame.USEREVENT + 2, 3000, True)
            return

        now = datetime.now(); timestamp = now.strftime("%Y%m%d_%H%M%S")
        grid_name_safe = "".join(c if c.isalnum() else "_" for c in self.selected_grid_key)
        agent_name_safe = "".join(c if c.isalnum() else "_" for c in self.selected_agent_type)
        filename = f"coop_chicken_metrics_{grid_name_safe}_{agent_name_safe}_{timestamp}.csv"

        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Overall Summary Metrics"]); writer.writerow(["Metric", "Value"])
                writer.writerow(["Agent Type", self.selected_agent_type])
                writer.writerow(["Grid Configuration", self.selected_grid_key])
                writer.writerow(["Episodes Run", self.dashboard_metrics.get("num_episodes", "N/A")])
                writer.writerow(["Avg. Reward Agent 1", f"{self.dashboard_metrics.get('avg_r1', 0):.2f}"])
                writer.writerow(["Avg. Reward Agent 2", f"{self.dashboard_metrics.get('avg_r2', 0):.2f}"])
                writer.writerow(["Avg. Episode Length (Rounds)", f"{self.dashboard_metrics.get('avg_len', 0):.2f}"])
                writer.writerow(["Total Captures", self.dashboard_metrics.get("captures", "N/A")])
                writer.writerow(["Capture Rate", f"{self.dashboard_metrics.get('capture_rate', 0):.2%}"])
                writer.writerow([]) 
                writer.writerow(["Per-Episode Details"])
                writer.writerow(["Episode No.", "Agent 1 Reward", "Agent 2 Reward", "Length (Rounds)", "Capture Occurred"])
                for i, res in enumerate(self.experiment_results):
                    capture_occurred = "No"
                    if res.get("history") and res["history"][-1].get("terminated", False): capture_occurred = "Yes"
                    writer.writerow([i + 1, f"{res.get('r1', 0):.2f}", f"{res.get('r2', 0):.2f}", res.get('length', 0), capture_occurred])
            self.status_message = f"Exported to {filename}"; print(f"Metrics successfully exported to {filename}")
        except IOError as e:
            self.status_message = "Error exporting CSV."; print(f"Error exporting CSV: {e}")
        pygame.time.set_timer(pygame.USEREVENT + 2, 5000, True) # Display message for 5s


    def _run_experiments_logic(self):
        grid_params = PREDEFINED_GRIDS[self.selected_grid_key]
        self.experiment_results = []
        
        print(f"Starting experiments: Agent={self.selected_agent_type}, Grid={self.selected_grid_key} (L={grid_params['L']}, H={grid_params['H']}, MaxSteps={grid_params['max_steps']})")
        for i in range(10): # Run 10 experiments
            print(f"  Running episode {i+1}/10...")
            env = CooperativeChickenEnv(
                L=grid_params["L"], H=grid_params["H"],
                internal_wall_coords=grid_params["walls"],
                max_episode_steps=grid_params["max_steps"]
            )
            if self.selected_agent_type == "Random":
                r1, r2, length, history = run_random_agents_episode(env, render_episode_to_console=False)
                self.experiment_results.append({
                    "r1": r1, "r2": r2, "length": length, "history": history,
                    "L": env.L, "H": env.H, "walls": list(env.walls) # Store for replay
                })
            # else: Implement other agent types here
        
        self._calculate_dashboard_metrics()
        self._generate_game_replay_buttons()
        self.current_screen = "dashboard"
        self.status_message = ""
        print("Experiments finished.")


    def _calculate_dashboard_metrics(self):
        if not self.experiment_results:
            self.dashboard_metrics = {"error": "No results to display."}
            return

        num_episodes = len(self.experiment_results)
        self.dashboard_metrics["num_episodes"] = num_episodes
        self.dashboard_metrics["avg_r1"] = sum(res["r1"] for res in self.experiment_results) / num_episodes
        self.dashboard_metrics["avg_r2"] = sum(res["r2"] for res in self.experiment_results) / num_episodes
        self.dashboard_metrics["avg_len"] = sum(res["length"] for res in self.experiment_results) / num_episodes
        
        captures = 0
        for res in self.experiment_results:
            if res["history"] and res["history"][-1]["terminated"]: # Check last step for termination
                captures +=1
        self.dashboard_metrics["captures"] = captures
        self.dashboard_metrics["capture_rate"] = captures / num_episodes if num_episodes > 0 else 0
        self.dashboard_metrics["episode_rewards"] = [(res["r1"], res["r2"], res["length"]) for res in self.experiment_results]


    def _generate_game_replay_buttons(self):
        self.game_replay_buttons = []
        if not self.experiment_results: return
        
        start_x = SIDE_PANEL_WIDTH + 50
        start_y = 120 # Below dashboard title
        button_w, button_h = 180, 35
        padding_x, padding_y = 15, 10
        max_cols = 3 # Max buttons per row in the replay selection area

        for i in range(len(self.experiment_results)):
            btn_text = f"Replay Game {i+1}"
            row = i // max_cols
            col = i % max_cols
            btn_x = start_x + col * (button_w + padding_x)
            btn_y = start_y + row * (button_h + padding_y)
            self.game_replay_buttons.append(Button(btn_x, btn_y, button_w, button_h, btn_text, font=FONT_SMALL))


    def render(self):
        self.screen.fill(WHITE)
        if self.current_screen == "selection":
            self.draw_selection_screen()
        elif self.current_screen == "running_experiments":
            status_text_surf = FONT_MEDIUM.render(self.status_message, True, BLACK)
            self.screen.blit(status_text_surf, status_text_surf.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2)))
        elif self.current_screen == "dashboard":
            self.draw_dashboard_screen()
        elif self.current_screen == "replay":
            self.draw_replay_screen()
        
    def draw_selection_screen(self):
        title_surf = FONT_LARGE.render("Cooperative Chicken Game - Experiment Setup", True, BLACK)
        self.screen.blit(title_surf, title_surf.get_rect(centerx=SCREEN_WIDTH // 2, y=20))

        self.screen.blit(self.agent_title_surf, self.agent_title_rect)
        for agent_type in AGENT_TYPES:
            self.buttons[f"select_agent_{agent_type}"].draw(self.screen)

        self.screen.blit(self.grid_title_surf, self.grid_title_rect)
        for grid_name in PREDEFINED_GRIDS.keys():
            self.buttons[f"select_grid_{grid_name}"].draw(self.screen)

        self.buttons["run_experiments"].draw(self.screen)

    def draw_dashboard_screen(self):
        # Left Panel for Metrics
        pygame.draw.rect(self.screen, LIGHT_GREY, (0, 0, SIDE_PANEL_WIDTH, SCREEN_HEIGHT))
        title_surf = FONT_LARGE.render("Dashboard", True, BLACK)
        self.screen.blit(title_surf, title_surf.get_rect(centerx=SIDE_PANEL_WIDTH // 2, y=20))

        if self.dashboard_metrics and "error" not in self.dashboard_metrics:
            metrics_y_start = 70
            line_height = 25
            
            info_texts = [
                f"Agent Type: {self.selected_agent_type}",
                f"Grid: {self.selected_grid_key}",
                f"Episodes Run: {self.dashboard_metrics['num_episodes']}",
                f"Avg. Reward A1: {self.dashboard_metrics['avg_r1']:.2f}",
                f"Avg. Reward A2: {self.dashboard_metrics['avg_r2']:.2f}",
                f"Avg. Ep. Length: {self.dashboard_metrics['avg_len']:.2f} rounds",
                f"Total Captures: {self.dashboard_metrics['captures']}",
                f"Capture Rate: {self.dashboard_metrics['capture_rate']:.2%}",
            ]
            for i, m_text in enumerate(info_texts):
                surf = FONT_SMALL.render(m_text, True, BLACK)
                self.screen.blit(surf, (10, metrics_y_start + i * line_height))
            
            current_y = metrics_y_start + len(info_texts) * line_height + 20
            ep_rewards_title = FONT_MEDIUM.render("Episode Summaries (R1, R2, Len):", True, BLACK)
            self.screen.blit(ep_rewards_title, (10, current_y))
            current_y += 30
            
            for i, (r1, r2, length) in enumerate(self.dashboard_metrics.get("episode_rewards", [])):
                ep_sum_text = f"Ep {i+1}: ({r1:.0f}, {r2:.0f}, {length})" # Using .0f for integer-like display
                surf = FONT_SMALL.render(ep_sum_text, True, BLACK)
                self.screen.blit(surf, (10, current_y + i* (line_height-5) ))
                if current_y + i*(line_height-5) > SCREEN_HEIGHT - 80 : break 
        else:
            err_text = FONT_MEDIUM.render(self.dashboard_metrics.get("error", "No data."), True, RED)
            self.screen.blit(err_text, (10, 100))

        # Right Panel for Game Selection
        replay_title_surf = FONT_LARGE.render("Select Game to Replay", True, BLACK)
        self.screen.blit(replay_title_surf, (SIDE_PANEL_WIDTH + 50, 20)) # Adjust x for panel start
        for btn in self.game_replay_buttons:
            btn.draw(self.screen)

        self.buttons["back_to_selection"].draw(self.screen)
        self.buttons["export_csv"].draw(self.screen)

        if self.status_message: # Display status message
            is_success = "Exported to" in self.status_message or "successfully" in self.status_message
            color = STATUS_SUCCESS_COLOR if is_success else STATUS_ERROR_COLOR
            status_surf = FONT_SMALL.render(self.status_message, True, color)
            self.screen.blit(status_surf, status_surf.get_rect(centerx=SIDE_PANEL_WIDTH // 2, bottom=SCREEN_HEIGHT - 10))


    def draw_replay_screen(self):
        # Main Grid Area (Left)
        grid_bg_rect = pygame.Rect(0,0, GRID_AREA_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, WHITE, grid_bg_rect) # Clear grid area specifically

        # Info Panel (Right)
        info_panel_rect = pygame.Rect(GRID_AREA_WIDTH, 0, SIDE_PANEL_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, LIGHT_GREY, info_panel_rect)
        
        if self.replay_game_index == -1 or not self.experiment_results or \
           not (0 <= self.replay_game_index < len(self.experiment_results)):
            err_text = FONT_LARGE.render("Error: No game selected.", True, RED)
            self.screen.blit(err_text, err_text.get_rect(center=self.screen.get_rect().center))
            # Draw back button even on error
            btn_back_dash = self.buttons.get("back_to_dashboard") # Use .get for safety
            if btn_back_dash: btn_back_dash.draw(self.screen)
            return

        game_data = self.experiment_results[self.replay_game_index]
        history = game_data.get("history")
        if not history:
            err_text = FONT_MEDIUM.render("Error: Selected game has no history.", True, RED)
            self.screen.blit(err_text, (50,50))
            btn_back_dash = self.buttons.get("back_to_dashboard")
            if btn_back_dash: btn_back_dash.draw(self.screen)
            return

        current_step_data = history[self.replay_step_index]
        
        grid_L = game_data["L"]
        grid_H = game_data["H"]
        grid_walls = set(map(tuple, game_data["walls"]))

        cell_w = (GRID_AREA_WIDTH - 40) // grid_H # -40 for padding
        cell_h = (GRID_AREA_HEIGHT - 40) // grid_L# -40 for padding
        cell_size = min(cell_w, cell_h)
        
        grid_total_w = grid_H * cell_size
        grid_total_h = grid_L * cell_size
        grid_offset_x = (GRID_AREA_WIDTH - grid_total_w) // 2
        grid_offset_y = 30 

        for r_idx in range(grid_L):
            for c_idx in range(grid_H):
                cell_rect = pygame.Rect(grid_offset_x + c_idx * cell_size, grid_offset_y + r_idx * cell_size, cell_size, cell_size)
                pygame.draw.rect(self.screen, GREY, cell_rect, 1) 
                if (r_idx, c_idx) in grid_walls:
                    pygame.draw.rect(self.screen, WALL_COLOR, cell_rect.inflate(-2,-2)) # Slightly smaller fill
        
        entity_radius_ratio = 0.35 # Ratio of cell_size
        font_size_in_cell = int(cell_size * 0.5)
        cell_font = pygame.font.Font(None, font_size_in_cell if font_size_in_cell > 10 else 12)


        def draw_entity(pos, color, label_text):
            if pos:
                center_x = grid_offset_x + pos[1] * cell_size + cell_size // 2
                center_y = grid_offset_y + pos[0] * cell_size + cell_size // 2
                pygame.draw.circle(self.screen, color, (center_x, center_y), int(cell_size * entity_radius_ratio))
                if label_text:
                    text_surf = cell_font.render(label_text, True, BLACK)
                    self.screen.blit(text_surf, text_surf.get_rect(center=(center_x, center_y)))
        
        draw_entity(current_step_data.get('agent1_pos'), AGENT1_COLOR, "1")
        draw_entity(current_step_data.get('agent2_pos'), AGENT2_COLOR, "2")
        draw_entity(current_step_data.get('chicken_pos'), CHICKEN_COLOR, "C")
        
        # --- Info Panel Content (Right) ---
        info_panel_x_start = GRID_AREA_WIDTH + 15
        title_surf = FONT_MEDIUM.render(f"Replay: Game {self.replay_game_index + 1}", True, BLACK)
        self.screen.blit(title_surf, (info_panel_x_start, 20))
        
        step_info_y = 60
        step_text_surf = FONT_SMALL.render(f"Step: {self.replay_step_index + 1} / {self.max_replay_steps}", True, BLACK)
        self.screen.blit(step_text_surf, (info_panel_x_start, step_info_y))
        step_info_y += 25
        
        acting_agent_val = current_step_data.get('acting_agent')
        acting_agent_str = "N/A"
        if acting_agent_val is not None:
            if acting_agent_val == -1 : acting_agent_str = "Initial/End"
            elif current_step_data.get('action_taken') is None: acting_agent_str = "Initial" # First step has no actor yet
            else: acting_agent_str = f"Agent {acting_agent_val + 1}"


        details_to_display = [
            f"Round No: {current_step_data.get('round_step', 'N/A')}",
            f"Current Turn: {acting_agent_str}",
            f"Action Taken: {current_step_data.get('action_taken', 'N/A')}",
            f"Reward This Step: {current_step_data.get('reward_received', 'N/A'):.1f}",
            f"A1 Total Reward: {current_step_data.get('total_reward_agent1_so_far', 'N/A'):.1f}",
            f"A2 Total Reward: {current_step_data.get('total_reward_agent2_so_far', 'N/A'):.1f}",
            f"Terminated: {current_step_data.get('terminated', 'N/A')}",
            f"Truncated: {current_step_data.get('truncated', 'N/A')}",
        ]
        for i, d_text in enumerate(details_to_display):
            surf = FONT_SMALL.render(d_text, True, BLACK)
            self.screen.blit(surf, (info_panel_x_start, step_info_y + i * 20))
        
        # --- Timeline & Controls (Bottom of Grid Area) ---
        timeline_base_y = SCREEN_HEIGHT - 90
        self.timeline_rect = pygame.Rect(50, timeline_base_y , GRID_AREA_WIDTH - 100, 15) # Update rect for correct pos
        pygame.draw.rect(self.screen, GREY, self.timeline_rect, border_radius=3)
        if self.max_replay_steps > 1:
            progress_percent = self.replay_step_index / (self.max_replay_steps -1)
            handle_pos_x = self.timeline_rect.x + int(progress_percent * self.timeline_rect.width)
            pygame.draw.circle(self.screen, BLUE, (handle_pos_x, self.timeline_rect.centery), 8)

        # Reposition buttons relative to timeline or fixed positions
        self.buttons["prev_step"].rect.midtop = (self.timeline_rect.centerx - 60, self.timeline_rect.bottom + 10)
        self.buttons["next_step"].rect.midtop = (self.timeline_rect.centerx + 60, self.timeline_rect.bottom + 10)
        self.buttons["prev_step"].draw(self.screen)
        self.buttons["next_step"].draw(self.screen)
        
        # Back to dashboard button is on the info panel side typically, adjust if needed
        self.buttons["back_to_dashboard"].rect.bottomleft = (GRID_AREA_WIDTH + 20, SCREEN_HEIGHT - 20)
        self.buttons["back_to_dashboard"].draw(self.screen)


if __name__ == '__main__':
    if not os.path.exists("environment.py"):
        print("ERROR: environment.py not found. Make sure it's in the same directory as visualizer.py.")
        sys.exit(1)
    if not os.path.exists("random_agents.py"):
        print("ERROR: random_agents.py not found. Make sure it's in the same directory as visualizer.py.")
        sys.exit(1)
        
    app = VisualizerApp()
    app.run()