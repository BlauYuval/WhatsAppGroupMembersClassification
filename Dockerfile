FROM pytorch/pytorch:latest


ENV CODE_DIR /opt/code/

WORKDIR /opt/code

RUN apt-get update && \ 
	apt-get install -y gcc g++ && \
	pip install --no-cache-dir jupyter && \
	pip install transformers&& \
	pip install transformers[torch]&&\
	pip install pandas&& \
	pip install matplotlib&& \
	pip install sentencepiece&& \ 
	pip install emoji&& \
	pip install sacremoses&& \
	pip install tqdm&& \
	pip install -U scikit-learn&& \
	mkdir -p ${CODE_DIR} 
	
WORKDIR ${CODE_DIR}

EXPOSE 8888
	
ENTRYPOINT ["jupyter", "notebook", "--ip='*'", "--NotebookApp.token=''", "--NotebookApp.password=''", "--allow-root"]