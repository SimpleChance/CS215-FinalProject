"""
Render class and methods for Travelling Salesman:
"""
import pygame as pg
from sys import exit


def event_listen():
    if not pg.get_init():
        pg.init()
    for e in pg.event.get():
        if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
            pg.quit()
            exit()


class Renderer(object):
    def __init__(self, display_dimensions, dimensions, max_framerate, scale=0.5, fullscreen=False,
                 colors=None, node_radius=5, path_width=2, border_width=3, start_ind=0,
                 end_ind=0):
        if not pg.get_init():
            pg.init()
            pg.display.init()
            pg.display.set_caption('Travelling Salesman Visualization')

        if fullscreen:
            display_dimensions = pg.display.get_desktop_sizes()[0]
            self.screen = pg.display.set_mode(display_dimensions)
            pg.display.toggle_fullscreen()
        else:
            self.screen = pg.display.set_mode(display_dimensions)

        self.clock = pg.time.Clock()
        self.header_font = pg.font.SysFont('Comic Sans MS', 20)
        self.text_font = pg.font.SysFont('Comic Sans MS', 15)
        self.max_framerate = max_framerate
        self.display_dimensions = display_dimensions

        self.scale = scale
        self.dimensions = dimensions
        self.node_to_display_scale_x = (display_dimensions[0] / dimensions[0]) * scale
        self.node_to_display_scale_y = (display_dimensions[1] / dimensions[1]) * scale
        self.display_offset = [(dimensions[0] * self.node_to_display_scale_x) // 2,
                               (dimensions[1] * self.node_to_display_scale_y) // 2]
        self.node_space_offset = [0, -100]

        self.node_radius = node_radius
        self.path_width = path_width
        self.border_width = border_width

        if colors is None:
            colors = [pg.Color(255, 255, 255), pg.Color(0, 0, 0), pg.Color(255, 0, 0),
                      pg.Color(0, 255, 0), pg.Color(125, 125, 255)]
        else:
            colors = [pg.Color(colors[0]), pg.Color(colors[1]), pg.Color(colors[2]),
                      pg.Color(colors[3]), pg.Color(colors[4])]

        self.background_color = colors[0]
        self.border_color = colors[1]
        self.start_color = colors[2]
        self.node_color = colors[3]
        self.path_color = colors[4]

        self.start_ind = start_ind
        self.end_ind = end_ind

    def update(self):
        self.clock.tick(self.max_framerate)
        pg.display.flip()

    def conv_to_render_coords(self, node):
        return [(node[0] * self.node_to_display_scale_x) + self.display_offset[0] + self.node_space_offset[0],
                (node[1] * self.node_to_display_scale_y) + self.display_offset[1] + self.node_space_offset[1]]

    def draw_background(self):
        self.screen.fill(self.background_color)

    def draw_node_space_border(self):
        x, y = self.conv_to_render_coords([0, 0])
        x -= self.node_radius*4
        y -= self.node_radius*4
        l, w = self.conv_to_render_coords([self.dimensions[0], self.dimensions[1]])
        l -= (self.display_offset[0] + self.node_space_offset[0])
        w -= (self.display_offset[1] + self.node_space_offset[1])
        l += self.node_radius*8
        w += self.node_radius*8
        pg.draw.rect(self.screen, self.border_color, (x, y, l, w), width=self.border_width)

    def draw_nodes(self, coords):
        i = 0
        for node in coords:
            render_coords = self.conv_to_render_coords(node)
            if i == self.start_ind or i == self.end_ind:
                pg.draw.circle(self.screen, self.start_color, render_coords, self.node_radius)
            else:
                pg.draw.circle(self.screen, self.node_color, render_coords, self.node_radius)
            i += 1

    def draw_path(self, path, coords, color=None):
        if color is None:
            color = self.path_color
        for i in range(len(path)-1):
            start_pos = self.conv_to_render_coords(coords[path[i]])
            end_pos = self.conv_to_render_coords(coords[path[i+1]])
            pg.draw.line(self.screen, color, start_pos, end_pos, self.path_width)

    def draw_header(self, header, pos):
        header_surface = self.header_font.render(header, False, (0, 0, 0))
        self.screen.blit(header_surface, pos)

    def draw_text(self, text, pos):
        text_surface = self.text_font.render(text, False, (0, 0, 0))
        self.screen.blit(text_surface, pos)

    def draw_frame(self, coords, path, text, opt_path=None):
        self.draw_background()
        self.draw_node_space_border()
        self.draw_nodes(coords)
        if opt_path:
            self.draw_path(opt_path, coords, (255, 200, 200))
            self.draw_path(path, coords)
        else:
            self.draw_path(path, coords)

        self.draw_header(f"Best Fit:",
                         [self.display_dimensions[0]//5 + 15, 3*self.display_dimensions[1]//5 + 10])
        self.draw_text(text[1], (self.display_dimensions[1]//5 + 45, 3*self.display_dimensions[1]//5 + 35))
        self.update()
