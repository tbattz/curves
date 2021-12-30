import numpy as np
import matplotlib.pyplot as plt


# Based on guide in https://pomax.github.io/bezierinfo/#control


class BezierCurve:
    """
    Creates and stores values to draw a bezier curve of order n given control points.
    """
    def __init__(self, ctrPts, n, ptWeights=None):
        self.n = n
        self.ctrPts = np.array(ctrPts)
        self.dim = self.ctrPts.shape[1]
        self.ptWeights = ptWeights

        self.xyz = []

        # Generate pascals triangle for the order n
        self.lut = self.genPascalsTriangle(k=n)


    def genPascalsTriangle(self, k=6, printTriangle=False):
        """
        Generate pascals triangle for order k.
        """
        lut = []
        if k > -1:
            lut.append([1])
        if k > 0:
            lut.append([1, 1])
        if k > 1:
            for i in range(1, k):
                row = [1]
                for j in range(0, len(lut[-1]) - 1):
                    row.append(lut[-1][j] + lut[-1][j + 1])
                row.append(1)
                lut.append(row)

        if printTriangle:
            for row in lut:
                print(row)

        return lut

    def binomial(self, n, k):
        """
        Compute the binomial using Pascal's triangle.
        Choose k from n.
        """
        return self.lut[n][k]


    def genStandardBezierValues(self, t):
        """
        Generate the point values for the curve at the given values of t.
        Bezier(n,t) = sum_i=0^n Bi(n,i)*(1-t)^(n-1)*t^i*w_i

        :param t: Is a list of paramterised values from 0 to 1.
        """
        # t is an array of values from 0 to 1
        sumXYZ = [0]*self.dim
        t = np.array(t)

        for k in range(0, self.n+1):
            leadingTerm = self.binomial(self.n, k) * np.power(1-t, self.n-k) * np.power(t,k)
            for i in range(0, self.dim):
                sumXYZ[i] += leadingTerm * self.ctrPts[k][i]

        self.xyz = sumXYZ

        return self.xyz


    def genRationalBezierValues(self, t):
        """
        Rational bezier curves - add an additional weighted ratio term for each point
        Rational Bezier(n,t) = (sum_i=0^n Bi(n,i)*(1-t)^(n-i)*t^i *w_i * ratio_i) / (sum_i=0^n Bi(n,i)*(1-t)^(n-i)*t^i * ratio_i)
        Where ratio is the point weights.

        :param t: An array of values from 0 to 1
        """
        ratioSum = np.zeros((1,t.shape[0]))
        weightSum = np.zeros((self.dim,t.shape[0]))
        for k in range(0, self.n+1):
            leadingTerm = self.binomial(self.n,k) * np.power(1-t, self.n-k) * np.power(t,k)
            ratioTerm = leadingTerm * self.ptWeights[k]
            for i in range(0, self.dim):
                weightTerm = ratioTerm * self.ctrPts[k][i]
                weightSum[i] += weightTerm

            ratioSum += ratioTerm

        self.xyz = weightSum/ratioSum

        return self.xyz


    def genBezierDeCasteljauAlgo(self, t, pts=None):
        """
        Uses de Casteljau's algorithm to generate the value for the bezier curve defined by points pts, for an array of t values.
        The first call of this function requites pts=None, and each recursive call afterwards will use the output of newPoints.
        self.ctrPts is of shape (n, d), where n is the number of control points, and d is the number of dimensions. e.g. For x,y, d=2.

        :param t: A array of t parameters values.
        :param pts: Set to None for first use. An array of shape (t, n, d), where t is the number of t values, n is the number of control points, and d the dimension.
        :return: An array of points at each t value, with shape (t, d)
        """
        if pts is None:
            # First use, create numpy array from control points
            pts = np.array(self.ctrPts)
        if len(pts.shape) != 3:
            # Tile points to an array of (t, n, d). This allows broadcasting to calculate all xyz and all t at the same time.
            pts = np.tile(pts, (t.shape[0], 1, 1))

        if pts.shape[1] == 1:
            # Only one point left, final values for each t value.
            val = pts.reshape(pts.shape[0], pts.shape[2])

            self.xyz = val
        else:
            # Calculate new points at t positions along lines
            t2 = t[:, np.newaxis, np.newaxis]
            newPoints = ((1-t2)*pts[:,:-1,:]) + (t2*pts[:,1:,:])

            val = self.genBezierDeCasteljauAlgo(t, pts=newPoints)

        return val

    def genBezierMatrix(self):
        """
        Generate the middle matrix of binomail coefficients for the matrix approach to generating a bezier curve.
        """
        coeff = [[self.binomial(self.n, i) * self.binomial(i, k) * (-1) ** (i - k) for k in range(i + 1)] for i in
                 range(self.n + 1)]

        # Pad with zeros to create square matrix
        return np.matrix([row + [0] * (self.n + 1 - len(row)) for row in coeff])

    def genBezierMatrixValues(self, t):
        """
        Generate the bezier curve values using a matrix approach.
        For a third order bezier curve.
        B(t) = [1 t t^2 t^3] * [[1 0 0 0], [-3 3 0 0], [3 -6 3 0], [-1 3 -3 1]]* [P1 P2 P3 P4]
        Where the coefficients for the middle matrix come from the binomial representation.

        :return: Points array of shape (t, d) where t is the number of paramter points and d is the dimension, e.g. for x, y pairs, d = 2.
        """
        # Calculate the binomial coeff matrix
        coeffs = self.genBezierMatrix()
        xyzMatrix = coeffs*self.ctrPts[:,:].reshape(self.ctrPts.shape[0],1,-1)

        # Generate the vector of t powers, [t^0, t^1, t^2, ...]
        tVec = np.power(t[:, np.newaxis, np.newaxis], np.arange(0,self.n+1))

        # Calculate the final point values.
        xyzm = tVec * xyzMatrix

        self.xyz = xyzm

        return xyzm

if __name__ == '__main__':
    # Create points
    pts = np.array([[110, 150],
           [25, 190],
           [210, 250],
           [210, 30]])

    # Create point weights for rational bezier curves
    ratio = [1.0, 0.5, 1.5, 1.0]

    # Create curve class and calculate points
    b = BezierCurve(pts, n=3, ptWeights=ratio)
    ta = np.linspace(0,1,100)
    xv, yv = b.genStandardBezierValues(ta)
    xrv, yrv = b.genRationalBezierValues(ta)
    xyzdcv = b.genBezierDeCasteljauAlgo(ta)
    xyzm2 = b.genBezierMatrixValues(ta)

    # Plot curves
    plt.plot(pts[:,0], pts[:,1], 'ro-', label='Control Points')
    plt.plot(xv, yv, 'k--', label='Bezier Class (Binomial)')
    plt.plot(xrv, yrv, 'c--', label='Bezier Class (Rational)')
    plt.plot(xyzdcv[:,0], xyzdcv[:,1], 'g--', label='Bezier Class (de Casteljau)')
    plt.plot(xyzm2[:,0], xyzm2[:,1], 'm--', label='Bezier Class (Matrix)')

    plt.legend()
    plt.show()


