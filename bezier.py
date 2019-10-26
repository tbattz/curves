import numpy as np


class BezierCubic:
	"""
	Calculates the cubic bezier curve for the provided points.
	"""
	def __init__(self, pts):
		"""
		:param pts: A numpy matrix of [[x0, x1, x2, x3], [y0, y1, y2, y3]]
		"""
		self.pts = pts

	def calcValue(self, t):
		"""
		Calculates the x, y values for the curve, from ths given pts at the specified value of t.
		C_0(t) = (1-t)^3 * P_0 + 3t(1-t)^2 * P_1 + 3t^2(1-t)*P_2 + t^3 * P_3

		:param t: The parameter value, between 0 and 1 inclusive.
		"""
		x = ((1 - t) ** 3 * pts[0][0]) + (3 * t * (1 - t) ** 2 * pts[0][1]) + (3 * (1 - t) * t ** 2 * pts[0][2]) + (
				t ** 3 * pts[0][3])
		y = ((1 - t) ** 3 * pts[1][0]) + (3 * t * (1 - t) ** 2 * pts[1][1]) + (3 * (1 - t) * t ** 2 * pts[1][2]) + (
					t ** 3 * pts[1][3])

		return x, y

class BezierNth:
	"""
	Calculates the nth degree bezier curve for the provided n+1 points.
	"""

	def __init__(self, pts):
		"""
		:param pts: A numpy matrix of [[x0, x1, ...], [y0, y1, ...]]
		"""
		self.pts = pts
		self.n = self.pts.shape[1] - 1

	def calcValue(self, t):
		"""
		Calculates the x, y values for the curve, from ths given pts at the specified value of t.
		Each nth degree is the linear combination of the n-1th degree polynomials.
		L2_0(t) = (1-t)*L1_0(t) + tL1_1(t)

		:param t: The parameter value, between 0 and 1 inclusive.
		"""
		currVals = self.pts
		for n in range(1, self.n + 1):
			nthVals = np.empty((2, currVals.shape[1] - 1))
			# Loop over groupings of points for this degree
			for i in range(0, nthVals.shape[1]):
				nthVals[:, i] = ((1 - t) * currVals[:, i]) + (t * currVals[:, i + 1])
			# Reassign the current values
			currVals = np.copy(nthVals)

		return currVals



# Point Matrix
#pts = np.array([[-7.0, 0.0, 7.0, 15.0], [5.0, -5.0, 10.0, 0.0]])
pts = np.array([[-7.0, 0.0, 7.0, 15.0, 17.0, 20.0, 5.0], [5.0, -5.0, 10.0, 0.0, -2.0, 3.0, 8.0]])

# Create Bezier Curve
bez = BezierCubic(pts)
bezNth = BezierNth(pts)

# Calculate points
xs = []
ys = []
xs2 = []
ys2 = []
for t in np.linspace(0, 1, 100):
	x, y = bez.calcValue(t)
	xs.append(x)
	ys.append(y)

	x2, y2 = bezNth.calcValue(t)
	xs2.append(x2)
	ys2.append(y2)

# Plot points
import matplotlib.pyplot as plt
plt.plot(xs, ys, 'bo-')
plt.plot(xs2, ys2, 'go-')
for i in range(0, len(pts)):
	plt.plot(pts[0], pts[1], 'ro')
plt.show()
