from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Box
from minigrid.minigrid_env import MiniGridEnv
from gymnasium.utils import seeding
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


class SimpleFoodBox(WorldObj):
    """
    Box with 2 states: 
        0: empty
        1: full
    """

    def __init__(self, color, contains: WorldObj | None = None):
        super().__init__("box", color)
        self.state = 0 # initially empty
    
    def pickup(self, env, pos):
        # if full, eat
        if self.state == 1:
            self.state = 0
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
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))
        
        # draw "food" if full
        if self.state == 1:
            fill_coords(img, point_in_circle(0.5, 0.5, 0.2), c)
            return

class EnergyBoxesEnv(MiniGridEnv):

    def __init__(
        self,
        size=5,
        agent_start_pos=(1,1),
        agent_start_dir=0,
        max_steps: int | None = None,
        refill_prob=0.1,
        initial_energy=10,
        time_bonus=0.1,
        box_open_reward=0,
        box_energy_refuel=8,
        seed=0,
        track_timestep_counts=False,
        **kwargs,
    ):  
        
        # set seed
        self.seed = seed
        self.np_random, _ = seeding.np_random(self.seed)

        self.width = size
        self.height = size
        
        # set up initial positions
        self.box_positions = [(1, self.height-2), (self.width-2, 1)]
        self.start_pos_random = agent_start_pos == "random"
        self.start_dir_random = agent_start_dir == "random"

        self.agent_start_pos = self._rand_pos() if self.start_pos_random else agent_start_pos
        self.agent_start_dir = self._rand_dir() if self.start_dir_random else agent_start_dir
        
        # box and goal dynamics
        self.refill_prob = refill_prob
        mission_space = MissionSpace(mission_func=self._gen_mission)
        if max_steps is None: max_steps = 512
        
        # energy initialization
        self.initial_energy = initial_energy
        self.energy = initial_energy
        self.time_energy_cost = 1
        self.action_energy_cost = 0 # TODO
        self.box_energy_refuel = box_energy_refuel
        self.time_bonus = time_bonus
        self.box_open_reward = box_open_reward

        # stats tracking
        self.eat_count = 0
        self.red_count = 0
        self.blue_count = 0
        self.previous_agent_pos = self.agent_start_pos
        self.agent_distance = 0
        self.last_box_opened = None
        self.consecutive_boxes = 0
        self.track_timestep_counts = track_timestep_counts
        self.timestep_counts = np.zeros(max_steps)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    def _rand_pos(self):
        pos = self.np_random.choice([(1,1), (self.width-2, self.height-2)])
        return pos
    
    def _rand_dir(self):
        return self.np_random.integers(0, 4)

    @staticmethod
    def _gen_mission():
        return "eat boxes to survive"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the two boxes
        self.grid.set(*self.box_positions[0], SimpleFoodBox(COLOR_NAMES[0])) # blue
        self.grid.set(*self.box_positions[1], SimpleFoodBox(COLOR_NAMES[4])) # red

        # Put food in one of the boxes at random
        if self.np_random.uniform() < 0.5:
            self.grid.get(*self.box_positions[0]).state = 1
        else:
            self.grid.get(*self.box_positions[1]).state = 1

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = EnergyBoxesEnv._gen_mission()

    def reset(self, **kwargs):

        #self.np_random, self.seed = seeding.np_random(self.seed) # reset seed
        obs = super().reset()

        self.agent_pos = self._rand_pos() if self.start_pos_random else self.agent_start_pos
        self.agent_dir = self._rand_dir() if self.start_dir_random else self.agent_start_dir

        self.previous_agent_pos = self.agent_start_pos
        self.energy = self.initial_energy

        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # give reward if forward cell is a full box and action is pickup (eat)
        fwd_cell = self.grid.get(*self.front_pos)
        if action == self.actions.pickup:
            fwd_cell = self.grid.get(*self.front_pos)
            if fwd_cell is not None and fwd_cell.type == "box":
                if fwd_cell.state == 1:
                    reward += self.box_open_reward
                    self.energy += self.box_energy_refuel
                    self.eat_count += 1
                    fwd_cell.state = 0

                    # stats tracking
                    if self.track_timestep_counts: self.timestep_counts[self.step_count-1] += 1
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
            if box.state == 0: # empty -> full with some probability
                if self.np_random.uniform() < self.refill_prob:
                    box.state = 1
        
        # energy dynamics
        self.energy -= self.time_energy_cost
        self.energy -= self.action_energy_cost
        
        if self.energy <= 0:
            terminated = True
            reward -= self.initial_energy * self.time_bonus
            self.reset()
        
        reward += self.time_bonus # reward TODO
        
        self.agent_distance += np.round(np.linalg.norm(np.array(self.agent_pos) - np.array(self.previous_agent_pos)))
        self.previous_agent_pos = self.agent_pos

        # return dict before resetting stats
        info['eat_count'] = self.eat_count
        info['red_count'] = self.red_count
        info['blue_count'] = self.blue_count
        info['agent_distance'] = self.agent_distance
        info['consecutive_boxes'] = self.consecutive_boxes
        info['mix_rate'] = 1.0 - (self.consecutive_boxes + 1 / self.eat_count) if self.eat_count > 0 else 0.0
        if self.track_timestep_counts: info['timestep_counts'] = self.timestep_counts
        
        # reset stats if episode ended
        if truncated or terminated:
            self.eat_count = 0
            self.red_count = 0
            self.blue_count = 0
            self.agent_distance = 0   
            self.consecutive_boxes = 0
            self.previous_agent_pos = self.agent_start_pos

        return obs, reward, terminated, truncated, info


class EnergyBoxesHardEnv(EnergyBoxesEnv):

    def __init__(self, **kwargs):

        super().__init__(
            size=5,
            initial_energy=5,
            refill_prob=0,
            box_energy_refuel=6,
            **kwargs,
        )
    
    def _gen_grid(self, width, height):

        super()._gen_grid(width, height)

        # set both boxes full
        self.grid.get(*self.box_positions[0]).state = 1
        self.grid.get(*self.box_positions[1]).state = 1

    def step(self, action):

        obs, reward, terminated, truncated, info = super().step(action)

        # refill opposite box if the other is empty and last box open was this one
        if self.grid.get(*self.box_positions[0]).state == 0 and self.last_box_opened == "blue":
            self.grid.get(*self.box_positions[1]).state = 1
        elif self.grid.get(*self.box_positions[1]).state == 0 and self.last_box_opened == "red":
            self.grid.get(*self.box_positions[0]).state = 1

        return obs, reward, terminated, truncated, info


class EnergyBoxesDelayEnv(EnergyBoxesEnv):
    """
    Subclass of EnergyBoxesEnv where the initial energy is high 
    so that the agent can survive for a long time without eating.
    """

    def __init__(self, **kwargs):
        super().__init__(
            size=6,
            initial_energy=100,
            refill_prob=0.05,
            box_energy_refuel=10,
            max_steps=512,
            **kwargs,
        )