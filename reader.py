import numpy as np
import matplotlib.pyplot as plt

# t = np.arange(0,8,1)
# c0 = [84.44004009675393, 89.44004009675393, 92.9050256302435]
# c0 = [105.25113690748789, 110.25113690748789, 235.78187387381502, 240.78187387381502, 410.99772039386966, 415.99772039386966, 404.88684136603945, 420.99772039386966, 437.03386409537546, 442.03386409537546, 1097.010426496439, 1102.010426496439, 1300.2258628312093, 1305.2258628312093, 1379.3328718918417, 1384.3328718918417, 1743.924963916404, 1748.924963916404, 1978.8784768558014, 1983.8784768558014, 2308.2595600651953, 2313.2595600651953, 2682.687947113056, 2687.687947113056, 3666.9352977975705, 3671.9352977975705, 3766.4771925687924]
# c0 =  [207.48725898892988, 400.12195910290626, 427.754686431356, 432.96473661015926, 454.21536327407154, 1906.3991271625885, 2888.702368096846, 3370.6939727124527, 3699.8041942440586, 3766.078100714719, 3766.714391665945, 3766.714391665945]
c0 = [95.76656349986953, 164.64988053111244, 205.4955444011511, 414.37901336415564, 437.4579390461092, 1137.340582870153, 1549.3864425267302, 1867.601182794225, 2540.899838828546, 2186.0861615452936, 2411.8637383140212, 2974.276512100289, 1990.9156098029682, 3082.690856162106, 3755.044415287656, 3766.714391665945, 3766.714391665945, 3766.714391665945, 3766.714391665945, 3766.714391665945, 3766.714391665945, 3766.714391665945, 3766.714391665945, 3766.714391665945, 3766.714391665945, 3766.714391665945, 3766.714391665945, 3766.714391665945, 3766.714391665945, 3766.714391665945, 3766.714391665945]
print(len(c0))
t = np.arange(0., len(c0), 1)

plt.plot(t, c0)
plt.gca().legend(('c0'))
plt.show()