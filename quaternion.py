import numpy as np

class quaternion(object):
    
    """
    Класс для представления кватерниона как 4-х мерного вектора в виде numpy массива
    """
    
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0, **kwargs):
        if kwargs:
            if ("scalar" in kwargs) or ("vector" in kwargs):
                scalar = kwargs.get("scalar", 0.0)
                if scalar is None:
                    scalar = 0.0
                else:
                    scalar = float(scalar)
                vector = kwargs.get("vector", [])
                vector = self._validate_number_sequence(vector, 3)
                self.w = scalar
                self.x = vector[0]
                self.y = vector[1]
                self.z = vector[2]
                self.q = np.hstack((scalar, vector))
            elif "array" in kwargs:
                self.q = self._validate_number_sequence(kwargs["array"], 4)
                self.w = self.q[0]
                self.x = self.q[1]
                self.y = self.q[2]
                self.z = self.q[3]
        else:
            # Инициилизация по умолчанию
            self.w = w
            self.x = x
            self.y = y
            self.z = z
            self.q = np.array([w, x, y, z])
    
    @classmethod
    def create(cls, axis, angle):
        """
        Классовый метод создания кватерниона вращения вектора X в трехмерном пространстве. 
        Действие сводится к вращению вектора X вокруг вектора axis на некоторый угол angle.
        Возвращает вектор, задающий новое положение в пространстве вектора X.
        arguments:
           axis {list / np.array} -- вектор-ось вращения;
           angle {int / float} -- угол поворота вокруг axis в радианах.
        returns:
           
        """
        axis = np.array(axis)
        axis_len = np.sqrt(np.dot(axis, axis))
        if axis_len == 0.0:
            raise ZeroDivisionError("Ось вращения не имеет длины: был задан нулевой вектор")
        if (abs(1.0 - axis_len) > 1e-12):
            axis = axis / axis_len # нормируем вектор оси вращения, если задан не единичный вектор
        q_sca = np.cos(angle / 2.0)
        q_vec = axis * np.sin(angle / 2.0)
        return cls(q_sca, q_vec[0], q_vec[1], q_vec[2])
    
    def rotate(self, vector):
        """
        Метод, поворачивающий вектор vector, с помощью поворота, заданного кватерионом q.
        """
        r = quaternion(vector=vector) # кватернион положения (0 + xi + yj + kz)
        self._normalise()
        r_new = self * r * self.conjugate # кватернион нового положения точки
        vector_new = r_new.vector
        return vector_new
    
    def get_axis(self, axis0=np.zeros(3)):
        tolerance = 1e-17
        self._normalise()
        norm = np.linalg.norm(self.vector)
        if norm < tolerance:
            return axis0
        else:
            return self.vector / norm
    
    @property
    def angle(self):
        self._normalise()
        norm = np.linalg.norm(self.vector)
        angl = 2.0 * np.arctan2(norm, self.scalar)
        angl = ((angl + np.pi) % (2 * np.pi)) - np.pi
        if angl == -np.pi:
            angl = np.pi
        return angl

    @property
    def axis(self):
        return self.get_axis()
    
    @property
    def scalar(self):
        return self.q[0]
    
    @property
    def vector(self):
        return self.q[1:]
    
    @property
    def real(self):
        return self.scalar

    @property
    def imaginary(self):
        return self.vector
    
    @property
    def norm(self):
        return np.sqrt(np.dot(self.q, self.q))
    
    @property
    def radians(self):
        return self.angle
    
    @property
    def degrees(self):
        return np.degrees(self.angle)
    
    @property
    def conjugate(self):
        """
        Свойство, возвращающее сопряженный кватернион (с отрицательной векторной частью)
        """
        return self.__class__(scalar=self.scalar, vector=-self.vector)
    
    def _normalise(self, tolerance=1e-14):
        """
        Метод нормализующий кватернион, если он таковым не является
        """
        if abs(1.0 - self.norm) > tolerance:
            if self.norm > 0:
                self.q = self.q / self.norm
    
    def _validate_number_sequence(self, seq, n):
        if seq is None:
            return np.zeros(n)
        if len(seq) == 0:
            return np.zeros(n)            
        elif len(seq) == n:
            try:
                l = [float(elem) for elem in seq]
            except ValueError:
                raise ValueError("Один или несколько элементов в последовательности <{!r}> не действительное(-ые) число(-а)".format(seq))
            else:
                return np.array(l)    
        else:
            raise ValueError("Неожиданное число элементов в последовательности. Получено: {}, Ожидалось: {}.".format(len(seq), n))
    
    def _q_matrix(self):
        """
        Представление кватерниона в виде матрицы для операции умножения
        """
        return np.array([
            [self.q[0], -self.q[1], -self.q[2], -self.q[3]],
            [self.q[1],  self.q[0], -self.q[3],  self.q[2]],
            [self.q[2],  self.q[3],  self.q[0], -self.q[1]],
            [self.q[3], -self.q[2],  self.q[1],  self.q[0]]
        ])
    
    def __repr__(self):
        return "q = [{!r}; {!r}; {!r}; {!r}]".format(self.q[0], self.q[1], self.q[2], self.q[3])
    
    def __str__(self):
        return "{:.3f}{:+.3f}i{:+.3f}j{:+.3f}k".format(self.q[0], self.q[1], self.q[2], self.q[3])
    
    def __int__(self):
        return int(self.q[0])
    
    def __float__(self):
        return float(self.q[0])
    
    def __getitem__(self, index):
        index = int(index)
        return self.q[index]
  
    def __mul__(self, other):
        if isinstance(other, quaternion):
            return self.__class__(array=np.dot(self._q_matrix(), other.q))

