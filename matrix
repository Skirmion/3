import mph
import numpy as np
import math 
import matplotlib.pyplot as plt
import sys

class matrix():
    def __init__(self, axis_direction, file_name, param_main_name, coor_center, coor_size, coor_size_min, current_start, names, xlen, ylen):
        self.axis_direction = axis_direction.lower()
        self.main = model([file_name, param_main_name, coor_center, coor_size], current_start)
        self.data = np.empty((xlen, ylen), dtype=object)
        self.Data_Filling(names, file_name, coor_size_min)

    def Data_Filling(self, names, file_name, coor_size_min):
        for i in names.keys():
            self.data[names[i][0][0], names[i][0][1]] = model([file_name, i, names[i][1], coor_size_min])

    def Obtain_Ness_B(self, mod):
        if self.axis_direction == "x":
            Bi = mod.Bx
        elif self.axis_direction == "y":
            Bi = mod.By
        elif self.axis_direction == "z":
            Bi = mod.Bz
        else:
            print("Ошибка: Указана некорректная ось")
            sys.exit()

    def B_Average_Vector(self, mod):
        Bi = self.Obtain_Ness_B(mod)
        buf = []
        for i in range(len(Bi)):
            buf += [Bi[i]]
        return np.sum(buf)/len(buf)
    
    def Define_Comparison_El(self):
        c = None
        c_compare = None
        init = np.empty(2)
        for i in range(len(np.array(self.data).shape[0])):
            for j in range(len(np.array(self.data).shape[1])):
                self.data[i][j].Run()
                c = self.B_Average_Vector(self.data[i][j])
                if c_compare == None or c_compare > c:
                    c_compare = c
                    init = [i, j]
        return init

    def Fitting(self):
        init = self.Define_Comparison_El()
        for i in range(len(np.array(self.data).shape[0])):
            for j in range(len(np.array(self.data).shape[1])):
        


#крч для начала надо брать не B-norm, а именно нужную компоненту вектора В
#затем сначала разбираемся с мелкими элементами для однородности, а потом уже аннигилируем общей переменной
#для каждого мелкого считаем среднее поле в своей области
#а затем рассматриваем разницу между этими средними для каждой пары мелких
#устраняем анизотропию
#профит

#тут я уже нашел ячейку с наименьшим полем и буду приводить все остальный к этой, а потом уже устранять анизотропию глобально
#надо научиться запускать фитинг для отдельной ячейки



    

class coords():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def Delete(self, i):
        self.x = np.delete(self.x, i, 0)
        self.y = np.delete(self.y, i, 0)
        self.z = np.delete(self.z, i, 0)


class model():
    def __init__(self, params, Current):
        self.file_name = params[0]
        self.name_param = params[1]
        self.zero_coords = params[2]
        self.size_coords = params[3]
        self.Current = Current
        self.obrazets, self.coords, self.B, self.B_average, self.B_max, self.B_min, self.B_rmse = None, None, None, None, None, None, None
        self.Bx, self.By, self.Bz = None, None, None

    def Run(self):
        self.Load_Model()
        self.Solve()
        self.Average()
        self.RMSE()

    def Load_Model(self):
        client = mph.start()
        self.obrazets = client.load(self.file_name)

    def Solve(self):
        #тут я обрашаюсь к глобальному параметру CurrentX, который создал сам
        self.obrazets.parameter(self.name_param, str(self.Current))
        self.obrazets.solve()
        xs, ys, zs, b, bx, by, bz = self.obrazets.evaluate(["x", "y", "z", "mfnc.normB - mfco.normB", "mfnc.Bx - mfco.Bx", "mfnc.By - mfco.By", "mfnc.Bz - mfco.Bz"])
        self.coords, self.B, self.Bx, self.By, self.Bz = self.Into_Area(coords(xs, ys, zs), self.zero_coords, self.size_coords, b, bx, by, bz)

    def Average(self):
        buf = []
        B_max = None
        B_min = None
        for i in range(len(self.B)):
            buf += [self.B[i]]
            if B_max == None or abs(B_max) < abs(self.B[i]):
                B_max = self.B[i]
            if B_min == None or abs(B_min) > abs(self.B[i]):
                B_min = self.B[i]
        self.B_average, self.B_max, self.B_min = np.sum(buf)/len(buf), B_max, B_min
        
    def RMSE(self):
        buf = []
        for i in range(len(self.B)):
                buf += [(self.B[i] - self.B_average)**2]
        self.B_rmse = math.sqrt(np.sum(buf)/len(buf))
    

    def Histogramm(self):
        ybuf = np.zeros(9)
        xbuf = [10**(-12+i) for i in range(10)]
        #xbuf = [self.B_min*st**(k+1) for k in range(10)]
        xxbuf = [str(10**(-12+i)) for i in range(10)]
        for i in range(len(self.B)):
                for j in range(len(xbuf) - 1):
                    if abs(self.B[i]) <= xbuf[j+1] and abs(self.B[i]) > xbuf[j]:
                        ybuf[j] += 1

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot()
        ax.bar(xxbuf[:9], ybuf)
        plt.show()

    def Into_Area_bool(self, coor1, coor2, coor_size):
        if abs(coor1.x - coor2.x) <= coor_size.x/2 and abs(coor1.y - coor2.y) <= coor_size.y/2 and abs(coor1.z - coor2.z) <= coor_size.z/2:
            return True
        else:
            return False
        
    def Into_Area(self, coor1, coor2, coor_size, B, Bx, By, Bz):
        p = 0
        for i in range(len(B)):
            #if abs(self.coords.x[i] - self.zero_coords.x) > self.size_coords.x/2 or abs(self.coords.y[i] - self.zero_coords.y) > self.size_coords.y/2 or abs(self.coords.z[i] - self.zero_coords.z) > self.size_coords.z/2:
            if abs(coor1.x[i-p] - coor2.x) > coor_size.x/2 or abs(coor1.y[i-p] - coor2.y) > coor_size.y/2 or abs(coor1.z[i-p] - coor2.z) > coor_size.z/2:
                B = np.delete(B, i-p, 0)
                Bx = np.delete(Bx, i-p, 0)
                By = np.delete(By, i-p, 0)
                Bz = np.delete(Bz, i-p, 0)
                coor1.Delete(i-p)
                p+=1
        return coor1, B, Bx, By, Bz
                
        
    def Area_Applic(self):
        xbuf = [10**(-12+i) for i in range(10)]
        coord_max = np.zeros((9, 3))
        xxbuf = [str(10**(-12+i)) for i in range(9)]
        for j in range(len(xbuf) - 1):
            for i in range(len(self.B)):
                if abs(self.B[i]) < xbuf[j+1]:
                        if abs(self.coords.x[i] - self.zero_coords.x) > abs(coord_max[j][0]):
                            coord_max[j][0] = abs(self.coords.x[i] - self.zero_coords.x)
                        if abs(self.coords.y[i] - self.zero_coords.y) > abs(coord_max[j][1]):
                            coord_max[j][1] = abs(self.coords.y[i] - self.zero_coords.y)
                        if abs(self.coords.z[i] - self.zero_coords.z) > abs(coord_max[j][2]):
                            coord_max[j][2] = abs(self.coords.z[i] - self.zero_coords.z)
        dicti = {str(xxbuf[g]) : 8*coord_max[g][0]*coord_max[g][1]*coord_max[g][2]/(self.size_coords.x*self.size_coords.y*self.size_coords.z) for g in range(len(xxbuf))}
        print("area application =", dicti)
        

def Prefitting(mod, step):
    params = [mod.file_name, mod.name_param, mod.zero_coords, mod.size_coords]
    current = mod.Current
    M1 = model(params, current + step)
    M1.Run()
    M2 = model(params, current - step)
    M2.Run()
    if abs(M1.B_average) > abs(M2.B_average):
        step = step*-1
        print("Prefitting:", current + step, M2.B_average)
        current += step
        M1.B_average = M2.B_average
    else:
        print("Prefitting:", current + step, M1.B_average)
        current += step
    while True:
        M3 = model(params, current + step)
        M3.Run()
        print("Prefitting:", current + step, M3.B_average)
        if abs(M3.B_average) > abs(M1.B_average):
            return current
        else:
            M1.B_average = M3.B_average
            current += step


def FittingC(mod, number_sim_after_comma, step):
    params = [mod.file_name, mod.name_param, mod.zero_coords, mod.size_coords]
    current_start = mod.Current
    print("Start at current =", current_start, "step =", step, "decimal point precision =", number_sim_after_comma)
    current_start = Prefitting(current_start, step)
    B_best = None
    C_best = None
    while 10**(-1*(number_sim_after_comma+1)) < step:
        M1 = model(params, current_start + step)
        M1.Run()
        M2 = model(params, current_start - step)
        M2.Run()

        if abs(M1.B_average) > abs(M2.B_average):
            current_start -= step
            B_present = M2.B_average
            M1.B_max = M2.B_max
        else:
            current_start += step
            B_present = M1.B_average
        step = step/2

        print("Fitting", current_start, B_present, M1.B_max)
        
        if B_best == None or abs(B_best) > abs(B_present):
            C_best = current_start
            B_best = B_present

    return C_best, B_best


#def B_vector(B):
    

#def Fitting_Matrix(params, matri, number_sim_after_comma, current_start, step):
#    current, B = FittingC(params, number_sim_after_comma, current_start, step)








#центр для пары квадратов
#x0 = 2.075
#y0 = 2.025
#z0 = 4.075
#центр для седел
#x0 = 0
#y0 = 0
#z0 = -0.04
#центр для двух окружностей
#x0 = 0
#y0 = 2.02
#z0 = 2.25
#центр для большого прямоугольника
x0 = 4
y0 = 2
z0 = 2
coor_center = coords(x0, y0, z0)
#размер экранируемой зоны:
x_zone = 2
y_zone = 2
z_zone = 4
coor_size = coords(x_zone, y_zone, z_zone)
#количество знаков после запятой
num_sim = 3

#file_name = "square.mph"
#param_name = "CurrentX"

params = ["helmholtzBIG.mph", "CurrentY", coor_center, coor_size]


current, B = FittingC(params, num_sim, 250, 20)
#current для седел = 179.69
M = model(params, round(current, num_sim))
M.Solve()
print("Итого:", "Current =",round(current, num_sim),"B average =", M.B_average,"B max =", M.B_max, "B RMSE =", M.B_rmse)
#M.histogramm()
#M.area_applic()
#cor = coords()
#print(cor.x)

# dic = {"Current1": [[0,0],[0.5, 0.5]], "Current2":[[0,1],[1.5, 1.5]]}
# m = matrix(dic, 1, 2)
# print(m.data)
