APP_NAME=ikibardin/spacenet5
CONTAINER_NAME=spcnt5_kek

# HELP
.PHONY: help

help: ## This help.
	@awk 'BEGIN (FS = ":.*?## ") /^[a-zA-Z_-]+:.*?## / (printf "\033[36m%-30s\033[0m %s\n", $$1, $$2)' $(MAKEFILE_LIST)

build:  ## Build the container
	nvidia-docker build -t $(APP_NAME) .


run-dl1: ## Run container at dl1
	nvidia-docker run \
		-itd \
		--ipc=host \
		--name=$(CONTAINER_NAME) \
		-v $(shell pwd):/code \
		-v /mnt/ssd1/dataset/spacenet5/nfs/data/cosmiq/spacenet/competitions/SN5_roads/tiles_upload:/data \
		-v /mnt/ssd1/dataset/spacenet5:/wdata \
		$(APP_NAME)

run-dl2: ## Run container at dl2
	nvidia-docker run \
		-itd \
		--ipc=host \
		--name=$(CONTAINER_NAME) \
		-v $(shell pwd):/code \
		-v /mnt/ssd2/dataset/spacenet5:/data \
		-v /mnt/ssd2/dataset/spacenet5:/wdata \
		$(APP_NAME)


run-dl4: ## Run container at dl4
	nvidia-docker run \
		-itd \
		--ipc=host \
		--name=$(CONTAINER_NAME) \
		-v $(shell pwd):/code \
		-v /mnt/ssd1/datasets/spacenet5:/data \
		-v /mnt/ssd1/datasets/spacenet5:/wdata \
		$(APP_NAME)

run-dl6: ## Run container at dl6
	nvidia-docker run \
		-itd \
		--ipc=host \
		--name=$(CONTAINER_NAME) \
		-v $(shell pwd):/code \
		-v /mnt/ssd1/dataset/spacenet5:/data \
		-v /mnt/ssd1/dataset/spacenet5:/wdata \
		$(APP_NAME)


run-ik: ## Run container at ik
	nvidia-docker run \
		-itd \
		--ipc=host \
		--name=$(CONTAINER_NAME) \
		-v $(shell pwd):/code \
		-v /home/ikibardin/dataset/spacenet5:/data \
		-v /home/ikibardin/dataset/spacenet5:/wdata \
		$(APP_NAME)


exec: ## Run a bash in a running container
	nvidia-docker exec -it $(CONTAINER_NAME) bash

stop: ## Stop and remove a running container
	docker stop $(CONTAINER_NAME); docker rm $(CONTAINER_NAME)