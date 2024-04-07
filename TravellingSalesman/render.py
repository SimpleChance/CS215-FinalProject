"""
Render class and methods for TravellingSalesman:
"""
import pygame as pg
from sys import exit


def event_listen():
    """
        Description: Listens for pygame events and determines if action should be taken
        Args: None
        Returns: void
    """
    if not pg.get_init():
        pg.init()
    for e in pg.event.get():
        if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
            pg.quit()
            exit()


class Renderer(object):
    """
        Description: Renderer class containing pygame methods and variables for rendering
        Args: list[int], list[int], int, colors=list[tuple(int)]
        Returns: void
    """

    def __init__(self, display_dimensions, dimensions, coordinates, max_framerate, scale=0.5, fullscreen=False,
                 colors=None, node_radius=5, path_width=2, border_width=3, start_ind=0,
                 end_ind=0):

        # Initialize pygame
        if not pg.get_init():
            pg.init()
            pg.display.init()
            pg.display.set_caption('TravellingSalesman Visualization')

        # Determine dimensions and update screen variable
        if fullscreen:
            display_dimensions = pg.display.get_desktop_sizes()[0]
            self.screen = pg.display.set_mode(display_dimensions)
            pg.display.toggle_fullscreen()
        else:
            self.screen = pg.display.set_mode(display_dimensions)

        # Font variables
        self.header_font = pg.font.SysFont('Comic Sans MS', 20)
        self.text_font = pg.font.SysFont('Comic Sans MS', 15)

        # Animation and display variables
        self.clock = pg.time.Clock()
        self.max_framerate = max_framerate
        self.display_dimensions = display_dimensions

        # Variables to scale and shift tsp dimensions to display dimensions
        self.scale = scale
        self.dimensions = dimensions
        self.node_to_display_scale_x = (display_dimensions[0] / dimensions[0]) * (scale * 1.5)
        self.node_to_display_scale_y = (display_dimensions[1] / dimensions[1]) * scale
        self.display_offset = [(dimensions[0] * self.node_to_display_scale_x) // 2 - int(self.display_dimensions[0] / 4),
                               (dimensions[1] * self.node_to_display_scale_y) // 2]
        self.node_space_offset = [0, -100]

        # Path and node rendering variables
        self.coords = []
        for node in coordinates:  # Precompute render coordinates for given nodes
            self.coords.append(self.conv_to_render_coords(node))
        self.node_radius = node_radius
        self.path_width = path_width
        self.border_width = border_width

        # Default colors
        if colors is None:
            colors = [pg.Color(255, 255, 255), pg.Color(0, 0, 0), pg.Color(255, 0, 0),
                      pg.Color(0, 255, 0), pg.Color(125, 125, 255), pg.Color(255, 125, 125)]
        else:
            colors = [pg.Color(colors[0]), pg.Color(colors[1]), pg.Color(colors[2]),
                      pg.Color(colors[3]), pg.Color(colors[4]), pg.Color(colors[5])]
        self.background_color = colors[0]
        self.border_color = colors[1]
        self.start_color = colors[2]
        self.node_color = colors[3]
        self.path_color = colors[4]
        self.opt_path_color = colors[5]

        # Start and end indices
        self.start_ind = start_ind
        self.end_ind = end_ind

        # Precompute border coordinates
        self.border_x, self.border_y = self.conv_to_render_coords([0, 0])
        self.border_x -= self.node_radius * 4
        self.border_y -= self.node_radius * 4
        self.border_l, self.border_w = self.conv_to_render_coords([self.dimensions[0], self.dimensions[1]])
        self.border_l -= (self.display_offset[0] + self.node_space_offset[0])
        self.border_w -= (self.display_offset[1] + self.node_space_offset[1])
        self.border_l += self.node_radius * 8
        self.border_w += self.node_radius * 8

    def update(self):
        """
            Description: Updates the clock and screen
            Args: None
            Returns: void
        """
        self.clock.tick(self.max_framerate)
        pg.display.flip()

    def conv_to_render_coords(self, node):
        """
            Description: Converts tsp node coordinates to display coordinates
            Args: list[int]
            Returns: list[int]
        """
        return [(node[0] * self.node_to_display_scale_x) + self.display_offset[0] + self.node_space_offset[0],
                (node[1] * self.node_to_display_scale_y) + self.display_offset[1] + self.node_space_offset[1]]

    def draw_background(self):
        """
            Description: Fills the screen with the background color
            Args: None
            Returns: void
        """
        self.screen.fill(self.background_color)

    def draw_node_space_border(self):
        """
            Description: Draws the border around the tsp instance
            Args: None
            Returns: void
        """
        pg.draw.rect(self.screen, self.border_color, (self.border_x, self.border_y, self.border_l, self.border_w),
                     width=self.border_width)

    def draw_nodes(self):
        """
            Description: Draws the nodes
            Args: None
            Returns: void
        """
        i = 0
        for node in self.coords:
            if i == self.start_ind or i == self.end_ind:  # If start or end node, use different color
                pg.draw.circle(self.screen, self.start_color, node, self.node_radius)
            else:
                pg.draw.circle(self.screen, self.node_color, node, self.node_radius)
            i += 1

    def draw_path(self, path, color=None):
        """
            Description: Draws the given path
            Args: list[int]
            Returns: void
        """
        # Default color
        if color is None:
            color = self.path_color

        # Draws a line between every node in order given by the path
        for i in range(len(path) - 1):
            start_pos = self.coords[path[i]]
            end_pos = self.coords[path[i + 1]]
            pg.draw.line(self.screen, color, start_pos, end_pos, self.path_width)

    def draw_header(self, header, pos):
        """
            Description: Draws the header text
            Args: string, list[int]
            Returns: void
        """
        header_surface = self.header_font.render(header, False, (0, 0, 0))
        self.screen.blit(header_surface, pos)

    def draw_text(self, text, pos):
        """
            Description: Draws the data text
            Args: string, list[int]
            Returns: void
        """
        text_surface = self.text_font.render(text, False, (0, 0, 0))
        self.screen.blit(text_surface, pos)

    def draw_frame(self, path, text, opt_path=None):
        """
            Description: Combines Renderer methods to draw an entire frame and update the screen
            Args: list[int], string
            Returns: void
        """
        # Draw background, node border, and nodes
        self.draw_background()
        self.draw_node_space_border()
        self.draw_nodes()

        # Find y location for text
        y = self.border_y + self.border_w
        x = self.border_x

        # Draws path and known optimum path if given
        if opt_path:
            # self.draw_path(opt_path, self.opt_path_color) # This doesn't look very good for now
            self.draw_path(path)
            self.draw_header(f"Known Optimum:",
                             [x + self.border_l - 135, y + 65])
            self.draw_text(text[-1],
                           (x + self.border_l - 135, y + 90))
        else:
            self.draw_path(path)

        # Draw text
        self.draw_header(f"Num Nodes:",
                         [x + self.border_l - 135, y + 10])
        self.draw_text(text[3],
                       (x + self.border_l - 135, y + 35))
        self.draw_header(f"Generation: ",
                         [x + 15, y + 10])
        self.draw_text(text[0], (x + 15, y + 35))
        self.draw_header(f"Elite Rate:",
                         [x + self.border_l // 2 - 50, y + 10])
        self.draw_text(text[5], (x + self.border_l // 2 - 50, y + 35))
        self.draw_header(f"Cross Rate:",
                         [x + self.border_l // 2 - 50, y + 65])
        self.draw_text(text[6], (x + self.border_l // 2 - 50, y + 90))
        self.draw_header(f"Mutation Rate:",
                         [x + self.border_l // 2 - 50, y + 120])
        self.draw_text(text[7], (x + self.border_l // 2 - 50, y + 145))
        self.draw_header(f"Population: ",
                         [x + 15, y + 65])
        self.draw_text(text[4], (x + 15, y + 90))
        self.draw_header(f"Best Fitness:",
                         [x + 15, y + 120])
        self.draw_text(text[1], (x + 15, y + 145))
        self.draw_header(f"Average Fitness:",
                         [x + 15, y + 175])
        self.draw_text(text[2], (x + 15, y + 200))

        # Update the screen
        self.update()
