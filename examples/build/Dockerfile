FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

LABEL maintainer="Jeremy Law <jjmlaw@gmail.com>"

ENV DEBIAN_FRONTEND=noninteractive

RUN set -x \
	&& apt-get update && apt-get install --no-install-recommends --no-install-suggests -y \
		build-essential \
		bzip2 \
		ca-certificates \
		cmake \
		curl \
		git \
		libjpeg-dev \
		libpng-dev \
		unzip \
		vim \
		zip \
	&& apt-get remove --auto-remove --purge -y \
	&& rm -rf /var/lib/apt/lists/*

ENV PYTHON_VERSION=3.6
RUN curl  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /var/tmp/miniconda.sh \
	&& chmod +x /var/tmp/miniconda.sh \
	&& /var/tmp/miniconda.sh -b -p /opt/conda \
	&& /opt/conda/bin/conda install conda-build -y \
	&& rm /var/tmp/miniconda.sh

WORKDIR /notebooks

COPY environment.yml /notebooks
COPY requirements.txt /notebooks

RUN set -x \
	&& /opt/conda/bin/conda env create \
	&& /opt/conda/bin/conda clean -ay

ENV PATH=$PATH:/opt/conda/envs/fastai/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
ENV USER fastai

CMD source activate fastai
CMD source ~/.bashrc

RUN set -x \
	&& /opt/conda/bin/conda install -c conda-forge -n fastai jupyterlab -y \
	&& chmod -R a+w /notebooks

COPY run.sh /run.sh

CMD ["/run.sh"]

#RUN set -x \
#	&& conda install -n fastai -y click \
#	&& conda update -c fastai -n fastai -y fastcore \
   # && conda install -c fastai -n fastai -y fastbook
