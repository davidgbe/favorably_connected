# hash:sha256:d21849f8ebb20ee0ee3c0af9e3948fbe1711d8eb03c9e87891faf50dcdd771de
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch:2.4.0-cuda12.4.0-mambaforge24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda install -y \
        gymnasium==0.26.3 \
        matplotlib==3.8.0 \
        numpy==1.26.0 \
        optuna==3.6.0 \
        python-blosc==1.11.2 \
        tqdm==4.67.0 \
    && conda clean -ya

COPY postInstall /
RUN /postInstall
