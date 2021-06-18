******
Docker
******


- download image from `Docker Hub <https://hub.docker.com/>`_

  .. code:: bash

     $ docker pull <image>   # name of the image to download

- list the images

  .. code:: bash

     $ docker images

- remove/delete an image

  .. code:: bash

     $ docker rmi <image>

- create a container from an image and start/run it

  .. code:: bash
  
     $ docker run -d \                       # run as a daemon
         -e <var>=<value> \                  # set environment variable <var> to <value> inside the container
         --mount \                           # mount a directory from the host system inside the container
             type=bind,\
             source=<host-dir>,\             # directory on the host system
             target=<container-dir> \        # directory on the container
         -p <host-port>:<container-port> \   # publish a container port on the host system
         -t \                                # 
         --name <name> \                     # name to give to the container
         <image>                             # image from which to create the container

- list the containers
  
  .. code:: bash

     $ docker ps         # list the currently running containers
     $ docker ps --all   # list all containers, including stopped ones

- stop/kill a running container

  .. code:: bash

     $ docker stop <container>   # the container name or ID
     $ docker kill <container>   # the container name or ID

- remove/delete a container

  .. code:: bash

     $ docker rm <container>

- run a command line command inside a container from the outside

  .. code:: bash

     $ docker exec <container> \   # the container name or ID
         <command> [<args>]        # the command (and possibly its arguments)

- create a new image from a container and its changes

  .. code:: bash

     $ docker commit \
         --change <change> \   # change to make to the container, e.g.
                               # - 'ENV <var> <value' to set/change an
                               #    environment variable
         <container> \         # the container to use as a basis/blueprint
         <image>               # the name of the image to create
