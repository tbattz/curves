import numpy as np
from stl import mesh

from os.path import expanduser

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl


# Load stl file
meshData = mesh.Mesh.from_file(expanduser('~') + '/Desktop/rotor/blade.stl')


# Plot points
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 20
w.show()
w.setWindowTitle('STL Points')

g = gl.GLGridItem()
w.addItem(g)

pos = meshData.points[:,0:3]
size = np.array([0.01]*len(pos))

scatterItem = gl.GLScatterPlotItem(pos=pos, size=size, pxMode=False)
w.addItem(scatterItem)


# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
	import sys
	if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
		QtGui.QApplication.instance().exec_()

