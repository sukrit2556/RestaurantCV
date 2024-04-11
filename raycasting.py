from polygenerator import random_polygon
import matplotlib.pyplot as plt


def is_inside(edges, xp, yp):
    cnt = 0
    for edge in edges:
        (x1, y1), (x2, y2) = edge
        if (yp < y1) != (yp < y2) and xp < x1 + ((yp-y1)/(y2-y1))*(x2-x1):
            cnt += 1
    return cnt%2 == 1


def onclick(event):
    xp, yp = event.xdata, event.ydata
    if is_inside(edges, xp, yp):
        print("inside")
        plt.plot(xp, yp, "go", markersize=5)
    else:
        print("outside")
        plt.plot(xp, yp, "ro", markersize=5)
    plt.gcf().canvas.draw()


#polygon = random_polygon(num_points=20)
polygon = [(90, 245), (318, 145), (427, 294), (148, 430)]
polygon.append(polygon[0])
edges = list(zip(polygon, polygon[1:]))
plt.figure(figsize=(10, 10))
plt.gca().set_aspect("equal")
plt.gca().invert_yaxis()
xs, ys = zip(*polygon)
plt.gcf().canvas.mpl_connect('button_press_event', onclick)
plt.plot(xs, ys, "b-", linewidth=0.8)
plt.show()
