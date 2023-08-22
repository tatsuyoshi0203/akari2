import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt


#ファイルの読み込み
with open('/content/drive/MyDrive/会議室1_19_53_16.txt', 'r') as file1:
  with open('/content/drive/MyDrive/会議室2_19_53_44.txt', 'r') as file2:

    source = file1.readlines()
    target = file2.readlines()

    data1 = []
    for line in source:
      row = [float(x) for x in line.split()]
      data1.append(row)

    data2 =[]
    for line in target:
      row2 = [float(x) for x in line.split()]
      data2.append(row2)

    matrix1 = np.array(data1)
    matrix2 = np.array(data2)

# 座標データのみを取り出す
Source = matrix1[:,:3]
Target = matrix2[:,:3]



#ソース点群とターゲット点群の対応付け
#kdtreeを使用
def sortxyz(arr:np.ndarray,axis_xyz:int,offset:int = 0):
    axis = axis_xyz + offset # offset
    return arr[np.argsort(arr[:,axis])]


x_col = 0
y_col = 1
z_col = 2
class Node:
    def set_node(self,points:np.ndarray,right:int,depth:int):
        if right < 0:
            return None
        elif right == 0:
            return self.set_leaf(points[right],depth)

        axis = depth % 3
        sorted_points = sortxyz(points[:right + 1],axis)
        if axis == 0:
            self.left_most = sorted_points[0][x_col]
            self.right_most = sorted_points[right][x_col]
        elif axis == 1:
            self.bottom_most = sorted_points[0][y_col]
            self.top_most = sorted_points[right][y_col]
        else:
            self.front_most = sorted_points[0][z_col]
            self.back_most = sorted_points[right][z_col]

        median = int(right / 2)
        self.border = (sorted_points[median][x_col:] + sorted_points[median+1][x_col:])/2.0
        self.location = sorted_points[median]
        self.depth = depth
        self.right_child = Node().set_node(sorted_points[median
                                                          + 1:],right -( median + 1),depth + 1)
        self.left_child = Node().set_node(sorted_points,median,depth + 1)

        #ここから先は関連付け
        if not(axis == 1):
            if(not(self.right_child is None) and not(self.left_child is None)):
                self.top_most = self.right_child.top_most if self.right_child.top_most > self.left_child.top_most else self.left_child.top_most
                self.bottom_most = self.right_child.bottom_most if self.right_child.bottom_most < self.left_child.bottom_most else self.left_child.bottom_most
            elif not(self.right_child is None):
                self.top_most = self.right_child.top_most
                self.bottom_most = self.right_child.bottom_most
            elif not(self.left_child is None):
                self.top_most = self.left_child.top_most
                self.bottom_most = self.left_child.bottom_most
            else:
                self.top_most = self.location[y_col]
                self.bottom_most = self.location[y_col]
        if not(axis == 0):
            if(not(self.right_child is None) and not(self.left_child is None)):
                self.right_most = self.right_child.right_most if self.right_child.right_most > self.left_child.right_most else self.left_child.right_most
                self.left_most = self.right_child.left_most if self.right_child.left_most < self.left_child.left_most else self.left_child.left_most
            elif not(self.right_child is None):
                self.right_most = self.right_child.right_most
                self.left_most = self.right_child.left_most
            elif not(self.left_child is None):
                self.right_most = self.left_child.right_most
                self.left_most = self.left_child.left_most
            else:
                self.right_most = self.location[x_col]
                self.left_most = self.location[x_col]
        if not(axis == 2):
            if(not(self.right_child is None) and not(self.left_child is None)):
                self.back_most = self.right_child.back_most if self.right_child.back_most > self.left_child.back_most else self.left_child.back_most
                self.front_most = self.right_child.front_most if self.right_child.front_most < self.left_child.front_most else self.left_child.front_most
            elif not(self.right_child is None):
                self.back_most = self.right_child.back_most
                self.front_most = self.right_child.front_most
            elif not(self.left_child is None):
                self.back_most = self.left_child.back_most
                self.front_most = self.left_child.front_most
            else:
                self.back_most = self.location[z_col]
                self.front_most = self.location[z_col]

        return self
    def set_leaf(self,location:np.ndarray,depth:int):
        self.location = location
        self.left_child = None
        self.right_child = None
        self.depth = depth
        self.left_most = location[x_col]
        self.right_most = location[x_col]
        self.top_most =   location[y_col]
        self.bottom_most = location[y_col]
        self.front_most = location[z_col]
        self.back_most = location[z_col]
        return self
    def is_contained(self,sx:int,tx:int,sy:int,ty:int,sz:int,tz:int):

        return not(self.left_most < sx or self.right_most > tx or \
                    self.top_most > ty or self.bottom_most < sy or \
                    self.front_most < sz or self.back_most > tz)
class Tree:
    def build(self,points):
        node = Node().set_node(points,len(points) - 1,0)
        self.top_node = node
        return node
    def search(self,r:list):
        sx,tx = r[0]
        sy,ty = r[1]
        sz,tz = r[2]
        search_results = []
        def _search(v:Node):
            nonlocal search_results,sx,tx,sy,ty
            if v.right_most < sx or v.left_most > tx or \
                v.bottom_most > ty or v.top_most < sy or \
                v.front_most > tz or v.back_most < sz:
                return;
            if v.left_child is None and v.right_child is None:
                if sx <= v.location[x_col] and sy <= v.location[y_col] and  sz <= v.location[z_col] and\
                    tx >= v.location[x_col] and ty >= v.location[y_col] and  tz >= v.location[z_col]:
                    self.search_results.append(v.location)
                    return
            if not(v.left_child is None):
                if(v.left_child.is_contained(sx,tx,sy,ty,sz,tz)):
                    search_results += self.report_subtree(v.left_child)
                else:
                    _search(v.left_child)
            if not(v.right_child is None):
                if(v.right_child.is_contained(sx,tx,sy,ty,sz,tz)):
                    search_results += self.report_subtree(v.right_child)
                else:
                    _search(v.right_child)
        _search(self.top_node)
        return search_results
    def report_subtree(self,node:Node = None,parent:Node=None,parent_border_axis:tuple = None,draw_border:tuple = None):
        """
        draw_border:tuple = (ax,max_x,min_x,max_y,min_y,max_z,min_z)
        """
        if node is None:
            node = self.top_node
        if node.left_child is None and node.right_child is None:
            return [node.location]
        pba = parent_border_axis
        if not (draw_border is None) :
            def _plot(axis:int,s,e):
                if axis == 0:
                    x = [s[0],e[0]]
                    y = [s[1],e[1]]
                    z = np.linspace(s[2], e[2])
                    Y, Z = np.meshgrid(y, z)
                    X = np.array([x] * Y.shape[0])
                    ax.plot_surface(X, Y, Z)
                elif axis == 1:
                    x = [s[0],e[0]]
                    y = [s[1],e[1]]
                    z = np.linspace(s[2], e[2])
                    X, Z = np.meshgrid(x, z)
                    Y = np.array([y] * X.shape[0])
                    ax.plot_surface(X, Y, Z)
                else:
                    x = np.linspace(s[0], e[0])
                    y = [s[1],e[1]]
                    z = [s[2],e[2]]
                    Z, X = np.meshgrid(z,x)
                    Y = np.array([y] * Z.shape[0])
                    ax.plot_surface(X, Y, Z)
            ax,xmax,xmin,ymax,ymin,zmax,zmin = draw_border
            s = node.border.copy()
            e = [0,0,0]
            if parent is None:
                s[1] = ymax
                s[2] = zmax
                e = np.array([s[0],ymin,zmin])
                pba = (s[0],None,s[2])
                _plot(0,s,e)
            else:
                if not(pba[0] is None) and not(pba[2] is None): #前がxの区切り(yz線) => 今がyの区切り(xz線)
                    x = xmax if(pba[0] < s[0]) else xmin
                    z = zmax if(pba[2] < s[2]) else zmin
                    s[0] = pba[0]
                    s[2] = pba[2]
                    e = np.array([x,s[1],z])
                    pba = (s[0],s[1],None)
                    _plot(1,s,e)
                elif not(pba[1] is None) and not(pba[0] is None): #前がyの区切り(xz線) => 今がzの区切り(xy線)

                    x = xmax if(pba[0] < s[0]) else xmin
                    y = ymax if(pba[1] < s[1]) else ymin
                    s[1] = pba[1]
                    s[0] = pba[0]
                    e = np.array([x,y,s[2]])
                    pba = (None,s[1],s[2])
                    _plot(2,s,e)
                else: #前がzの区切り(xy線) => 今がxの区切り(yz線)
                    y = ymax if(pba[1] < s[1]) else ymin
                    z = zmax if(pba[2] < s[2]) else zmin
                    s[2] = pba[2]
                    s[0] = pba[0]
                    e = np.array([s[0],y,z])
                    pba = (s[0],None,s[2])
                    _plot(0,s,e)

        arr = []
        if not (node.left_child is None):
            arr += self.report_subtree(node.left_child,node,pba,draw_border)
        if not (node.right_child is None):
            arr += self.report_subtree(node.right_child,node,pba,draw_border)
        return arr

points = Source
tree = Tree()
node = tree.build(points)
tree.report_subtree(draw_border=(ax,points[:,x_col].max(),points[:,x_col].min(),points[:,y_col].max(),points[:,y_col].min(),points[:,z_col].max(),points[:,z_col].min()))


tree = Tree()
node = tree.build(Source)
results = np.array(tree.search(Target))






#剛体変形の推定
#ターゲット点群とソース点群の重心を求める
#重心の計算
xaxis = 0            #重心のx座標
x_values1 = matrix1[:,xaxis]
x_average1 = np.mean(x_values1)

x_values2 = results[:,xaxis]
x_average2 = np.mean(x_values2)

yaxis = 1              #重心のy座標
y_values1 = matrix1[:,yaxis]
y_average1 = np.mean(y_values1)

y_values2= results[:,yaxis]
y_average2 = np.mean(y_values2)

zaxis = 2              #重心のz座標
z_values1 = matrix1[:,zaxis]
z_average1 = np.mean(z_values1)

z_values2 = results[:,zaxis]
z_average2 = np.mean(z_values2)

#点群の重心を原点に合わせる
Source[:,xaxis] -= x_average1
Source[:,yaxis] -= y_average1
Source[:,zaxis] -= z_average1

results[:,xaxis] -= x_average2
results[:,yaxis] -= y_average2
results[:,zaxis] -= z_average2

#回転行列を求める

#レジストレーションベクトルq の初期化と４元数から回転行列への変換
def quaternion2rotation( q ):
    rot = np.array([[q[0]**2+q[1]**2-q[2]**2-q[3]**2,
                     2.0*(q[1]*q[2]-q[0]*q[3]),
                     2.0*(q[1]*q[3]+q[0]*q[2])],

                    [2.0*(q[1]*q[2]+q[0]*q[3]),
                    q[0]**2+q[2]**2-q[1]**2-q[3]**2,
                     2.0*(q[2]*q[3]-q[0]*q[1])],

                    [2.0*(q[1]*q[3]-q[0]*q[2]),
                     2.0*(q[2]*q[3]+q[0]*q[1]),
                    q[0]**2+q[3]**2-q[1]**2-q[2]**2]]
                  )
    return rot

q = np.array([1.,0.,0.,0.,0.,0.,0.])
rot = quaternion2rotation(q)
print(rot)

rmse=[100,100,100]

while np.all(rmse > np.array((0.005,0.005,0.005))):   #rmseが閾値を下回るまで処理を繰り返す。
  #共分散の計算
  covar = np.zeros( (3,3) )
  n_points = Source.shape[0]
  for i in range(n_points):
     covar += np.dot( Source[i].reshape(-1, 1), results[i].reshape(1, -1) )
  covar /= n_points
  covar -= np.dot( x_average1.reshape(-1,1), y_average1.reshape(1,-1) )
  print(covar)

  #対象行列Npyを作成する
  A = covar - covar.T
  delta = np.array([A[1,2],A[2,0],A[0,1]])
  tr_covar = np.trace(covar)
  i3d = np.identity(3)

  N_py = np.zeros((4,4))
  N_py[0,0] = tr_covar
  N_py[0,1:4] = delta
  N_py[1:4,0] = delta
  N_py[1:4,1:4] = covar + covar.T - tr_covar*i3d
  print(N_py)

  #回転行列に変換する
  w, v = LA.eig(N_py)
  rot = quaternion2rotation(v[:,np.argmax(w)])
  print("Eigen value\n",w)
  print("Eigen vector\n",v)
  print("Rotation\n", rot)

  #物体の姿勢アップデート
  transform = np.identity(3)
  transform[0:3,0:3] = rot.copy()
  print("Transformation\n", transform)
  Source = np.dot(Source, transform.T)  # Sourceにtransformを適用して更新した行列をSourceに代入
  #RMSEを用いた精度評価
  # RMSEを計算する関数
  def calculate_rmse(predictions, true_values):
     n = len(predictions)
     squared_errors = [(pred - true) ** 2 for pred, true in zip(predictions, true_values)]
     mean_squared_error = sum(squared_errors) / n
     rmse = np.sqrt(mean_squared_error)
     return rmse

  # RMSEを計算
  rmse = calculate_rmse(Source, results)
  print("RMSE:", rmse)
