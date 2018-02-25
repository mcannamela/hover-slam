import matplotlib

print(matplotlib.get_configdir())
print(matplotlib.matplotlib_fname())
import matplotlib.pyplot as plt
import numpy as np
import os

t = np.arange(0.0, 2.0, 0.01)
s = np.sin(2 * np.pi * t)
plt.plot(t, s)

plt.xlabel('time (s)')
plt.ylabel('voltage (mV)')
plt.title('About as simple as it gets, folks')
plt.grid(True)
plt.savefig(os.path.join("resources", "matplotlib_tmp", "test.png"))
plt.show()
