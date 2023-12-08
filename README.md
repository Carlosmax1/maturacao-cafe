<p align=center>
<img  style="display: block; margin: 0 auto"  width=200px heigth=200px src="brasao.gif" alt="Brasão Universidade Federal de Viçosa" />
<h1 align=center>Universidade Federal de Viçosa</h1>
</p>
<p align=center><i>Campus</i> Rio Paranaíba<br> Curso: Sistemas de Informação <br> Disciplina SIN 323 - Inteligência Artificial
</p>
<br>

<h1 align=center>Técnicas de aprendizado de máquina aplicadas na classificação da maturação de grãos de café<br></h1>

<p>
<h1 align=center>Autores</h1>
<p align= center>
   Carlos Eduardo Maximo - 6962<br> Ronald Augusto Domingos Silva - 7024 <br> Augusto de Faria Pereira - 7556
</p>
</p>
<hr> <br>

<h3>1. Objetivo:</h3>
<p>O presente trabalho tem como objetivo desenvolver um sistema de visão para classificação de grãos de café entre maduros e verdes. Propõe-se a implementação de técnicas avançadas de processamento de imagens e aprendizado de máquina para avaliar a maturação dos grãos de café, buscando a classificação automática em "maduros" ou "imaturos". O modelo de aprendizado de máquina será baseado em técnicas de visão computacional.O objetivo central é fornecer uma ferramenta precisa e objetiva aos produtores para avaliar a qualidade dos grãos em tempo real. Isso contribuirá para otimizar o processo de seleção, melhorar a produção</p>

<h3>2. Como rodar o projeto: </h3>
<li>
O projeto foi feito com a linguagem de programação <strong>Python</strong> na sua versão <strong>3.8.10</strong>

<a href="https://www.python.org/downloads/release/python-3810/">Download Python</a>
</li>

<li>
  Após a instalação do Python em sua versão 3.8.10, faça o clone do repositório para sua máquina com o comando:
  <pre align=center>git clone https://github.com/Carlosmax1/maturacao-cafe.git</pre>
</li>

<li>
  Dentro a pasta onde foi clonado o repositório abra o terminal e digite o comando para instalar todas as dependências.
  <pre align=center>
    !pip install -r requirements.txt
  </pre>
</li>
<li>
Em seguida, rode o script <strong style="font-style: italic;">treatment.py</strong> para separação do dataset para treino e teste 
 <pre align=center>
    python3 treatment.py
  </pre>
</li>

<li>
  Após concluir os passos anteriores, basta executar o <strong style="font-style: italic;">main.py</strong> para treinar a IA. Após o treino, a pasta <span style="font-style:italic;">outputs</span> que haverá todos os relátorios do treinamento feito.
  <pre align=center>
    python3 main.py
  </pre>
</li>