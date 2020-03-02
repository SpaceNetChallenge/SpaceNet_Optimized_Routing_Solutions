build:  ## Build the container
	nvidia-docker build -t cresi docker

run:    ## Run container
	nvidia-docker run \
		-itd \
		--ipc=host \
		--rm \
		--name=cresi-container \
        -v $(shell pwd):/code \
		-v /mnt/ssd1/dataset/spacenet5:/data \
		cresi

exec: ## Run a bash in a running container
	nvidia-docker exec -it cresi-container bash


stop: ## Stop and remove a running container
	docker stop cresi-container