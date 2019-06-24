import pygame
from pygame.locals import *
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.signal import argrelextrema

carWidth,carLength,carNose=15,30,5
pontoCarro=[[-carWidth/2,-carLength/2],[carWidth/2,-carLength/2],[carWidth/2,carLength/2],[0,carLength/2+carNose],[-carWidth/2,carLength/2],
		[-4,carLength/2+4],[-24,carLength/2+30],[4,carLength/2+4],[24,carLength/2+30],[0,carLength/2+carNose+1],[0,carLength/2+carNose+33]]
#pontoCarro = [[-7,-15], [7,-15],[7,15],[0,20],[-7,15],[-4,19],[-24,45],[4,19],[24,45],[0,21],[0,48]]
linhaCarro=[[0,1],[1,2],[2,3],[3,4],[2,4],[0,4],[5,6],[7,8],[9,10]]
corCarro=[(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(0,255,0),(0,255,0),(0,255,0)]

# Function to find a local minima in an array 
  
# A binary search based function that  
# returns index of a local minima. 
# This code is contributed by Anant Agarwal.

def localMinUtil(arr, low, high, n): 
          
    # Find index of middle element 
    mid = low + (high - low) // 2  
          
    # Compare middle element with its  
    # neighbours (if neighbours exist) 
    if(mid == 0 or arr[mid - 1] > arr[mid] and
       mid == n - 1 or arr[mid] < arr[mid + 1]): 
        return mid 
          
    # If middle element is not minima and its left 
    # neighbour is smaller than it, then left half 
    # must have a local minima. 
    elif(mid > 0 and arr[mid - 1] < arr[mid]): 
        return localMinUtil(arr, low, mid - 1, n) 
          
    # If middle element is not minima and its right 
    # neighbour is smaller than it, then right half 
    # must have a local minima. 
    return localMinUtil(arr, mid + 1, high, n) 
      
# A wrapper over recursive function localMinUtil() 
def localMin(arr, n): 
      
    return localMinUtil(arr, 0, n - 1, n)                 

def calcularDist(A,B,C,D):
	min,max=np.minimum(A,B),np.maximum(A,B)
	if (C[0]<min[0] and D[0]<min[0]) or (C[1]<min[1] and D[1]<min[1]) or (C[0]>max[0] and D[0]>max[0]) or (C[1]>max[1] and D[1]>max[1]):
		return -1
	if np.dot(np.subtract(B,A),np.subtract(D,C))==0:
		if np.dot(np.subtract(B,A),np.subtract(C,A))==0:
			return 0
		return -1
	if C[0]==D[0]:
		t=(C[0]-A[0])/(B[0]-A[0])
		tp=(A[1]-t*(B[1]-A[1])-C[1])/(D[1]-C[1])
	else:
		k=B[1]-A[1]-(D[1]-C[1])*(B[0]-A[0])/(D[0]-C[0])
		t=(-A[1]+C[1]+(D[1]-C[1])*(A[0]-C[0])/(D[0]-C[0]))/k
		tp=(A[0]+t*(B[0]-A[0])-C[0])/(D[0]-C[0])
	if t>=0 and t<=1 and tp>=0 and tp<=1:
		return t
	return -1

def movimentoCarro(dx, dy, dtheta):
	sint = math.sin(dtheta)
	cost = math.cos(dtheta)
	return np.matrix([[cost,sint],[sint, -cost],[dx,dy]])


def renderisar(tela,showScreen, pontos, linhas, cores,transform=[]):
	if len(transform)>0:
		tPontos = []
		for ponto in pontos:
			p = [ponto[0],ponto[1],1]
			tPontos.extend(np.asarray(np.matmul(p,transform)))
	else:
		tPontos=pontos
	if(showScreen):
		for i in range(len(linhas)):
			pygame.draw.line(tela,cores[i],tuple(tPontos[linhas[i][0]]),tuple(tPontos[linhas[i][1]]))
	return tPontos

def monitorarB(tela, showScreen, rota, modelo):
	[modeloPontos, modeloLinhas,_] = modelo
	rotaPontos, rotaLinhas = rota
	batidaTemp = []

	for linh in modeloLinhas:
		aux = 2
		for rlinh in rotaLinhas:
			t = calcularDist(modeloPontos[linh[0]], modeloPontos[linh[1]], rotaPontos[rlinh[0]], rotaPontos[rlinh[1]])
			if t>=0 and t<=1:
				desenhoBatida=np.add(modeloPontos[linh[0]],np.multiply(t,np.subtract(modeloPontos[linh[1]], modeloPontos[linh[0]])))
				if(showScreen):
					pygame.draw.circle(tela, (255, 255, 0),(int(desenhoBatida[0]), int(desenhoBatida[1])),2)
					pygame.draw.line(tela,(255, 0, 255),rotaPontos[rlinh[0]], rotaPontos[rlinh[1]])
					#print("t = ", t)
				if t < aux:
					aux = t
		if aux == 2:
			batidaTemp.append(-1)
		else:
			batidaTemp.append(aux)
	return batidaTemp

def desenharCarro(tela, carro, showScreen):
	mat = movimentoCarro(carro['posX'],carro['posY'],carro['theta'])
	(points,lines,colors) = carro['modelo']
	return renderisar(tela, showScreen, points,lines,colors,transform=mat)

def desenhandoPath(tela, p0, p1, p2, p3, width = 40, line = 30, color = (0, 0, 255), points = [], lines = [], colors = []):
	offset=len(points)
	for t in np.arange(0, (1 + line)/line, 1/line):
		novoPonto = np.add(
			np.add(
				np.add(
					np.multiply((1-t)**3, p0),
					np.multiply(3*t*(1-t)**2, p1)
				),
				np.multiply(3*t**2*(1-t), p2)
			),
			np.multiply(t**3, p3)
		)

		speed = np.add(
			np.add(
				np.add(
					np.multiply(-3*t**2+6*t-3, p0),
					np.multiply(3*(3*t**2-4*t+1), p1)
				),
				np.multiply(
					3*(-3*t**2+2*t), p2)
					),
					np.multiply(3*t**2, p3)
		)
		length = math.sqrt(np.dot(speed, speed))
		vect = np.multiply(1/length, [speed[1], -speed[0]])
		points.append(tuple(np.add(novoPonto, np.multiply(width/2, vect))))
		points.append(tuple(np.add(novoPonto, np.multiply(-width/2, vect))))

	for i in range(line):
		lines.append([offset+i*2, offset+i*2+2])
		lines.append([offset+i*2+1, offset+i*2+3])
		colors.append(color)
		colors.append(color)

	return points, lines, colors

##OK
def mover(carro,dt):
	if carro['bateu']:
		return carro
	if carro['potencia']>1:
		carro['potencia']=1.0
	elif carro['potencia']<-1:
		carro['potencia']=-1.0
	if carro['giro']>0.03:
		carro['giro']=0.03
	elif carro['giro']<-0.03:
		carro['giro']=-0.03
	
	carro['velocidade']+= 30*carro['potencia']*dt-0.02*carro['velocidade']
	if carro['velocidade']<0.0:
		carro['velocidade']=0.0

	carro['posX']+=math.sin(carro['theta'])*carro['velocidade']*dt
	carro['posY']+=-math.cos(carro['theta'])*carro['velocidade']*dt
	carro['theta']+=carro['giro']*carro['velocidade']*dt
	carro['distancia']+=carro['velocidade']*dt

	return carro

def atualiza(carro):
	sensor=carro['sensor']
	cromosomo=carro['cromosomo']
	carro['potencia']=cromosomo[0]*sensor[0]+cromosomo[1]*sensor[1]+cromosomo[2]*sensor[2]+cromosomo[3]
	carro['giro']=cromosomo[4]/10*sensor[0]+cromosomo[5]/10*sensor[1]+cromosomo[6]/10*sensor[2]+cromosomo[7]/10
	return carro

def escolherCarros(numPop, carro):
	carrosTemp=[]
	for i in range(numPop):
		carrosTemp.append(carro.copy())
		carrosTemp[-1]['cromosomo']=0.1*np.random.randn(8)
	return carrosTemp

def mutacao(gene, ajuste):
	return gene + ajuste*np.random.randn(8)

def proximaGeracao(carros, carroMod, geracao, carrTemp):
	pontuacao = sorted(carros, key=lambda k: k['distancia'], reverse=True)
	filhos = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5]
	if geracao % 10 == 0:
		print("OKKK")
		filhos[len(filhos) - 1] = 15
		print (filhos)

	carrosTemp = []
	mutationRate = 0.05
	isOK = False
	arrayMin = carrTemp[:]
	print('arr = ', arrayMin)
	n = len(arrayMin)
	if(n > 0):
			maxElement = np.max(arrayMin)
			arrayMin.append(maxElement + 5)
			npArrayMin = np.array(arrayMin)
			minInd = argrelextrema(npArrayMin, np.less)
			r = npArrayMin[minInd]  # array([5, 3, 6])
			out_arr = r[np.nonzero(r)] 
			# print ("Output array of non-zero number: ", out_arr)
			isOK = True
		# print('isOK = ', isOK)
	if(n > 2 and arrayMin[n-2] in out_arr):
		mutationRate = 2 * 0.05
	else:
		mutationRate = 0.05
	print('mutRate = ', mutationRate)

	for i in filhos:
		carrosTemp.append(carroMod.copy()) 
		#print('n = ', n)
		
		# if(isOK):
		# 	if(n > 3 and out_arr[n-1] < min(arrayMin[n-2], arrayMin[n-3], arrayMin[n-4])):
		# 		mutationRate = 2 * 0.05
		# 	else:
		# 		mutationRate = 0.05
		carrosTemp[-1]['cromosomo'] = mutacao(pontuacao[i]['cromosomo'], mutationRate)
	return carrosTemp, pontuacao[0]['distancia']

def main():
	#Criando a janela
	pygame.init()
	tela = pygame.display.set_mode((640,480))
	font = pygame.font.SysFont('Arial', 20)
	pygame.display.set_caption('GA Labyrinto')

	t = 0
	dt = 0.1
	acidente = 0
	numPop = 16
	geracao = 1
	melPont = 0
	terminou = True
	vetPont = []
	showScreen = False 

	# Desenhando o caminho - de preferencia azul
	pontosPath, linhasPath, corPath = desenhandoPath(tela, [250,150], [350,150], [450,100], [450,200])
	pontosPath, linhasPath, corPath = desenhandoPath(tela, [450,200], [450,300], [300,200], [300,300], points = pontosPath, lines = linhasPath, colors = corPath)
	pontosPath, linhasPath, corPath = desenhandoPath(tela, [300,300], [300,500], [550,450], [550,300], points = pontosPath, lines = linhasPath, colors = corPath)
	pontosPath, linhasPath, corPath = desenhandoPath(tela, [550,300], [550,50], [550,50], [250,50], points = pontosPath, lines = linhasPath, colors = corPath)
	pontosPath, linhasPath, corPath = desenhandoPath(tela, [250,50], [50,50], [50,150], [50,350], points = pontosPath, lines = linhasPath, colors = corPath)
	pontosPath, linhasPath, corPath = desenhandoPath(tela, [50,350], [50,450], [150,450], [150,350], points = pontosPath, lines = linhasPath, colors = corPath)
	pontosPath, linhasPath, corPath = desenhandoPath(tela, [150,350], [150,250], [150,150], [250,150], points = pontosPath, lines = linhasPath, colors = corPath)

	carroInicial = {
		'posX' : 250.0,
		'posY' : 150.0,
		'theta' : 1.5,
		'velocidade' : 0.0, # Parado certo? 
		'giro' : 0.0, # giro limitado em +- 0.03
		'potencia' : 0.0, # limitado em +- 1.0 -- coeficiente que determina o quanto posso acelerar ou nao, ja que carro['velocidade']+=30*carro['potencia']*dt-0.02*carro['velocidade']
		'distancia' : 0,
		'modelo': [
					pontoCarro, 
					linhaCarro, 
					corCarro
				],
		'sensor' : [1.0, 1.0, 1.0], # retorna numero entre 0 e 1, quando é quer dizer que o sensor nao viu nada ainda
		'bateu' : False,
		'localmin': []
	}
	carrTemp = []

	carros=escolherCarros(numPop,carroInicial)
	y = {}
	begin=time.time()+dt
	while terminou:
		for event in pygame.event.get():
			if event.type == QUIT:
				terminou = False

		
		tela.fill((220, 220, 220))
		# if (geracao > 5):
		showScreen = True
		renderisar(tela, showScreen, pontosPath, linhasPath, corPath)

		for carro in carros:
			
			carroTela = desenharCarro(tela, carro, showScreen)
			if carro['bateu'] == False:
				bateu = monitorarB(tela, showScreen, (pontosPath, linhasPath), [carroTela, carro['modelo'][1], carro['modelo'][2]])
				end = False
				for i in range(6):
					if bateu[i] != -1 and carro['bateu'] == False:
						carro['bateu']=True
						acidente += 1
				for i in range(6,9):
					if bateu[i] == -1:
						bateu[i] = 1
				carro['sensor'] = bateu[6:]
				carro = atualiza(carro)
				carro = mover(carro,dt)

		if len(carros) > 0:
			t += dt
			if t > 30 or acidente >= numPop:
				t,acidente = 0,0
				print('geracao = ', geracao)
				carros, pontuacao = proximaGeracao(carros, carroInicial, geracao, carrTemp)
				carrTemp.append(pontuacao)
				print('pont = ', pontuacao)
				print('pont = ', carrTemp)
				if pontuacao>melPont:
					melPont=pontuacao
				# print(geracao)
				# print(melPont)
				#i = 0
				#for carro in carros:
					# print (f"c{i} = {list(carro['cromosomo'])}")
				#	i = i+1
				geracao+=1

		tela.blit(font.render('Geracao: {0}'.format(geracao), True, (100,0,0)),(0,0))
		tela.blit(font.render('Tempo: {0:.2f}s'.format(t), True, (100,0,0)),(0,25))
		tela.blit(font.render('Pontuacao: {0:.1f}'.format(melPont), True, (100,0,0)),(0,50))
		pygame.display.flip()
if __name__ == "__main__":
    main()