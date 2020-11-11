import matplotlib.pyplot as plt 

x = [1,2,5,6,7]
y = [2,3,7,1,0]
z = [200,500,100,3300,100]

titulo = "Gráfico de dispersão"
eixox = "Eixo do x "
eixoy = "Eixo do y"

plt.title(titulo)
plt.xlabel(eixox)
plt.ylabel(eixoy)

plt.scatter(x,y, label = "Grupo 1", color="k", marker=".", s=z, )#desenha o gráfico 
plt.plot(x,y, color="k", linestyle="--") #gráfico de linhas 
plt.legend() #mostrar as labels 
#plt.show
plt.savefig("graficodispersao.png", dpi=300)#guarda a figura, dpi aumenta a resolucao da figura 
plt.savefig("graficodispersao.pdf")

"""
color: cor (ver exemplos abaixo)
label: rótulo
linestyle: estilo de linha (ver exemplos abaixo)
linewidth: largura da linha
marker: marcador (ver exemplos abaixo)

'b' blue
'g' green
'r' red
'c' cyan
'm' magenta
'y' yellow
'k' black
'w' white

Marcadores (marker)
'.' point marker
',' pixel marker
'o' circle marker
'v' triangle_down marker
'^' triangle_up marker
'<' triangle_left marker
'>' triangle_right marker
'1' tri_down marker
'2' tri_up marker
'3' tri_left marker
'4' tri_right marker
's' square marker
'p' pentagon marker
'*' star marker
'h' hexagon1 marker
'H' hexagon2 marker
'+' plus marker
'x' x marker
'D' diamond marker
'd' thin_diamond marker
'|' vline marker
'_' hline marker

Tipos de linha (linestyle)
'-' solid line style
'--' dashed line style
'-.' dash-dot line style
':' dotted line style

Fonte: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html

"""