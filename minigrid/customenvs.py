from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Box
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import WorldObj, fill_coords, point_in_rect, point_in_line, COLORS
from minigrid.core.constants import (
    COLOR_TO_IDX,
    COLORS,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
)
from minigrid.utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_line,
    point_in_rect,
)
import numpy as np


class CustomBox(WorldObj):

    def __init__(self, color, contains: WorldObj | None = None):
        super().__init__("box", color)
        self.contains = contains
        self.state = 0 # 0: closed, 1: half-open, 2: open
        self.half_open_step = False

    def toggle(self, env, pos):
        # if closed, open
        if self.state == 0:
            self.state = 1
            self.half_open_step = True
            return True
        return False

    def can_pickup(self):
        return False
    
    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], self.state)

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.32, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.38, 0.82), (0, 0, 0))

        if self.state == 0:
            fill_coords(img, point_in_circle(0.5, 0.6, 0.1), c)
            return
            
        # draw open cover if half-open
        if self.state == 1:
            fill_coords(img, point_in_rect(0.18, 0.82, 0.32, 0.82), (0, 0, 0))
            fill_coords(img, point_in_line(0.17, 0.34, 0.85, 0.14, 0.03), c)
            fill_coords(img, point_in_circle(0.5, 0.6, 0.1), c)
            return

        # remove cover if open
        if self.state == 2:
            fill_coords(img, point_in_rect(0.18, 0.82, 0.32, 0.82), (0, 0, 0))

class BaseBoxesEnv(MiniGridEnv):

    """
    
    | Num | Name         | Action               |
    |-----|--------------|----------------------|
    | 0   | left         | Turn left            |
    | 1   | right        | Turn right           |
    | 2   | forward      | Move forward         |
    | 3   | pickup       | Eat                  |
    | 4   | drop         | Unused               |
    | 5   | toggle       | Open box             |
    | 6   | done         | Unused               |
    """

    def __init__(
        self,
        size=5,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        close_prob=0.1,
        **kwargs,
    ):
        # set up initial positions
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        
        # box and goal dynamics
        self.close_prob = close_prob
        mission_space = MissionSpace(mission_func=self._gen_mission)
        if max_steps is None:
            max_steps = max(256, size*size*5)

        # stats tracking
        self.eat_count = 0
        self.red_count = 0
        self.blue_count = 0
        self.previous_agent_pos = agent_start_pos
        self.agent_distance = 0
        self.last_box_opened = None
        self.consecutive_boxes = 0

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        raise NotImplementedError

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # give reward if forward cell is half-open box and action is pickup (eat)
        fwd_cell = self.grid.get(*self.front_pos)
        if action == self.actions.pickup:
            fwd_cell = self.grid.get(*self.front_pos)
            if fwd_cell is not None and fwd_cell.type == "box":
                if fwd_cell.state == 1:
                    reward += 1
                    self.eat_count += 1
                    if fwd_cell.color == "red":
                        self.red_count += 1
                        if self.last_box_opened == "red": self.consecutive_boxes += 1
                        else: self.last_box_opened = "red"
                    elif fwd_cell.color == "blue":
                        self.blue_count += 1
                        if self.last_box_opened == "blue": self.consecutive_boxes += 1
                        else: self.last_box_opened = "blue"

        # box dynamics
        for box_pos in self.box_positions:
            box = self.grid.get(*box_pos)
            if box.state == 1: # half-open -> open
                if box.half_open_step:
                    box.half_open_step = False
                else:
                    box.state = 2
            elif box.state == 2: # open -> closed with some probability
                if self.np_random.uniform() < self.close_prob:
                    box.state = 0
        
        self.agent_distance += np.round(np.linalg.norm(np.array(self.agent_pos) - np.array(self.previous_agent_pos)))
        self.previous_agent_pos = self.agent_pos

        # return dict before resetting stats
        info['eat_count'] = self.eat_count
        info['red_count'] = self.red_count
        info['blue_count'] = self.blue_count
        info['agent_distance'] = self.agent_distance
        info['consecutive_boxes'] = self.consecutive_boxes
        info['mix_rate'] = 1.0 - (self.consecutive_boxes / self.eat_count) if self.eat_count > 0 else 0.0
        
        # reset stats if episode ended
        if truncated or terminated:
            self.eat_count = 0
            self.red_count = 0
            self.blue_count = 0
            self.agent_distance = 0   
            self.consecutive_boxes = 0
            self.previous_agent_pos = self.agent_start_pos

        return obs, reward, terminated, truncated, info


class SimpleBoxesEnv(BaseBoxesEnv):

    def __init__(
        self,
        size=5,
        agent_start_pos=(2, 1),
        agent_start_dir=1,
        max_steps: int | None = None,
        close_prob=0.1,
        **kwargs,
    ):
        super().__init__(size, agent_start_pos, agent_start_dir, max_steps, close_prob, **kwargs)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Place the two boxes
        self.box_positions = [(1, height-2), (width-2, 1)]
        self.grid.set(*self.box_positions[0], CustomBox(COLOR_NAMES[0])) # blue
        self.grid.set(*self.box_positions[1], CustomBox(COLOR_NAMES[4])) # red

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = SimpleBoxesEnv._gen_mission()


class MazeBoxesEnv(BaseBoxesEnv):

    def __init__(
        self,
        size=9,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        close_prob=0.01,
        **kwargs,
    ):
        super().__init__(size, agent_start_pos, agent_start_dir, max_steps, close_prob, **kwargs)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate vertical and horizontal separation walls
        for i in range(2, height-3):
            self.grid.set(4, i, Wall())
        self.grid.set(4, height-2, Wall())

        for i in range(1, height-2):
            self.grid.set(i, 4, Wall())
        
        # Place the two boxes
        self.box_positions = [(1, height-3), (width-2, 2)]
        self.grid.set(*self.box_positions[0], CustomBox(COLOR_NAMES[0])) # blue
        self.grid.set(*self.box_positions[1], CustomBox(COLOR_NAMES[4])) # red

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = SimpleBoxesEnv._gen_mission()

class SwitchingBoxesEnv(MiniGridEnv):

    def __init__(
        self,
        size=6,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        close_prob=0.1,
        **kwargs,
    ):
        # set up initial positions
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        
        # box and goal dynamics
        self.close_prob = close_prob
        mission_space = MissionSpace(mission_func=self._gen_mission)
        if max_steps is None:
            max_steps = max(256, size*size*5)

        # stats tracking
        self.eat_count = 0
        self.red_count = 0
        self.blue_count = 0
        self.previous_agent_pos = agent_start_pos
        self.agent_distance = 0
        self.last_box_opened = None
        self.consecutive_boxes = 0

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the two boxes
        self.box_positions = [(1, height-2), (width-2, 1)]
        self.grid.set(*self.box_positions[0], CustomBox(COLOR_NAMES[0])) # blue
        self.grid.set(*self.box_positions[1], CustomBox(COLOR_NAMES[4])) # red

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = SwitchingBoxesEnv._gen_mission()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # give reward if forward cell is half-open box and action is pickup (eat)
        fwd_cell = self.grid.get(*self.front_pos)
        if action == self.actions.pickup:
            fwd_cell = self.grid.get(*self.front_pos)
            if fwd_cell is not None and fwd_cell.type == "box":
                if fwd_cell.state == 1:
                    reward += 1
                    self.eat_count += 1
                    if fwd_cell.color == "red":
                        self.red_count += 1
                        if self.last_box_opened == "red": self.consecutive_boxes += 1
                        else: self.last_box_opened = "red"
                    elif fwd_cell.color == "blue":
                        self.blue_count += 1
                        if self.last_box_opened == "blue": self.consecutive_boxes += 1
                        else: self.last_box_opened = "blue"

        # box dynamics
        for box_pos in self.box_positions:
            box = self.grid.get(*box_pos)
            if box.state == 1: # half-open -> open
                if box.half_open_step:
                    box.half_open_step = False
                else:
                    box.state = 2
            elif box.state == 2: # open -> closed with some probability (opposite color)
                if box.color != self.last_box_opened:
                    if self.np_random.uniform() < self.close_prob:
                        box.state = 0
        
        self.agent_distance += np.round(np.linalg.norm(np.array(self.agent_pos) - np.array(self.previous_agent_pos)))
        self.previous_agent_pos = self.agent_pos

        # return dict before resetting stats
        info['eat_count'] = self.eat_count
        info['red_count'] = self.red_count
        info['blue_count'] = self.blue_count
        info['agent_distance'] = self.agent_distance
        info['consecutive_boxes'] = self.consecutive_boxes
        info['mix_rate'] = 1.0 - (self.consecutive_boxes / self.eat_count) if self.eat_count > 0 else 0.0
        
        # reset stats if episode ended
        if truncated or terminated:
            self.eat_count = 0
            self.red_count = 0
            self.blue_count = 0
            self.agent_distance = 0   
            self.consecutive_boxes = 0
            self.previous_agent_pos = self.agent_start_pos

        return obs, reward, terminated, truncated, info