from turtle import *

def draw_square(): 
    i = 0
    while i < 4:
        forward(30)
        right(90)
        i += 1

def change_row():
    right(90)
    forward(30)
    left(90)

def change_column():
    forward(30)

def reset_row(distance_to_return):
    right(180)
    while distance_to_return > 0:
        forward(30)
        distance_to_return -= 1
    left(180)

def draw_row(number_of_squares):
    i = 0
    while i < number_of_squares:
        draw_square()
        change_column()
        i += 1
    reset_row(number_of_squares)
    change_row()

def go_to_top_left():
    left(90)
    forward(30)
    right(90)

draw_row(1)
draw_row(2)
draw_row(3)
draw_row(4)
draw_row(5)
draw_row(4)
draw_row(3)
draw_row(2)
draw_row(1)

input()
