#include <stdio.h>
#include <stdlib.h>
 
#define SRAND_VALUE 1985
// Iterate from 0 to dim, if you want to access ghost cells use -1 or dim as index
int* get(int dim, int* world, int y, int x) {
	return &world[(x+1) + (dim+2)*(y+1)];
}


void printWorld(int dim, int* world) {
  int i, j;
  printf("------\n");
  for(i = -1; i<=dim; i++) {
      for(j = -1; j<=dim; j++) {
        printf("%d", *get(dim, world, j, i));
      }
      printf("\n");
    }
}


void GOL_CPU(int dim, int *world, int *newWorld)
{
  int i, j, id;
  
  //Ghost row
  for(i = 0; i < dim; i++) {
    //TODO: Copy first real row to bottom ghost row
  	*get(dim, world, -1, i) = *get(dim, world, dim-1, i);

    //TODO: Copy last real row to top ghost row
    *get(dim, world, dim, i) = *get(dim, world, 0, i);
  }
  
  //Ghost column
  for(i = 0; i < dim; i++) {
    //TODO: Copy first real column to right most ghost column
    *get(dim, world, i, -1) = *get(dim, world, i, dim-1);
    //TODO: Copy last real column to left most ghost column
    *get(dim, world, i, dim) = *get(dim, world, i, 0);
  }

  // Corners
  *get(dim, world, -1, -1) = *get(dim, world, dim-1, dim-1);
  *get(dim, world, -1, dim) = *get(dim, world, 0, dim-1);
  *get(dim, world, dim, -1) = *get(dim, world, dim-1, 0);
  *get(dim, world, dim, dim) = *get(dim, world, 0, 0);

  printWorld(dim, world);

  
  for (i = 0; i < dim; i++) {
    for (j = 0; j < dim; j++) {      
      //world point
      int numNeighbors;

      // Get the number of neighbors for a world point
      numNeighbors = 	
      		*get(dim, world, i, j+1) 	//TODO: lower
  			+ *get(dim, world, i, j-1) 	//TODO: upper
  			+ *get(dim, world, i+1, j)	//TODO: right
  			+ *get(dim, world, i-1, j)	//TODO: left

  			+ *get(dim, world, i+1, j+1)	//TODO: diagonal lower right
  			+ *get(dim, world, i+1, j-1)	//TODO: diagonal upper right
  			+ *get(dim, world, i-1, j+1)	//TODO: diagonal lower left
  			+ *get(dim, world, i-1, j-1);	//TODO: diagonal upper left

      // game rules for Conways 23/3-world
      // 1) Any live cell with fewer than two live neighbours dies
      if (*get(dim, newWorld, i, j) && numNeighbors < 2)
		    *get(dim, newWorld, i, j) = 0;

      // 2) Any live cell with two or three live neighbours lives
      else if (*get(dim, newWorld, i, j) && (numNeighbors == 2 || numNeighbors == 3))//TODO
      	*get(dim, newWorld, i, j) = 1;

      // 3) Any live cell with more than three live neighbours dies
      else if (*get(dim, newWorld, i, j) && (numNeighbors > 3))//TODO
      	*get(dim, newWorld, i, j) = 0;

      // 4) Any dead cell with exactly three live neighbours becomes a live cell
      else if (!*get(dim, newWorld, i, j) && numNeighbors == 3)//TODO
      	*get(dim, newWorld, i, j) = 1;
      else
      	*get(dim, newWorld, i, j) = *get(dim, world, i, j);
    }
  }
}

void initRandom(int dim, int* world)
{
  int i, j;
 
  // Assign initial population randomly
  srand(SRAND_VALUE);
  for(i = 0; i<dim; i++) {
    for(j = 0; j<dim; j++) {
      *get(dim, world, j, i) = rand()%2;
    }
  }
}

void initGlider(int dim, int* world)
{
	int i, j;
	for(i = 0; i<dim; i++) {
	    for(j = 0; j<dim; j++) {
	      *get(dim, world, j, i) = 0;
	    }
  	}

  	*get(dim, world, 0, 1) = 1;
  	*get(dim, world, 1, 2) = 1;
  	*get(dim, world, 2, 0) = 1;
  	*get(dim, world, 2, 1) = 1;
  	*get(dim, world, 2, 2) = 1;
}


void create_PPM(int dim, int* world, FILE* fp)
{
  // Write header for ppm file (portable pixmap format)
  fprintf(fp, "P3\n");				// Portable Pixmap in ASCII encoding
  fprintf(fp, "%i %i\n", dim, dim);		// Dimension of the picture in pixel
  fprintf(fp, "255\n");				// Maximal color value
  
  // Sum cells and write world to file
  int total = 0;
  int i, j;
  
  for (i = 0; i<dim; i++) {
    for (j = 0; j<dim; j++) {
      if(*get(dim, world, i, j)) { 
		fprintf(fp, "255 255 255   ");		//dead cell is white
		total++;
	  } else {
		fprintf(fp, "  0   0    0   ");	//living cell is black
      }
      //calculate total cells alive
   
    }
    fprintf(fp, "\n");				// new column new line 
  }

  printf("Total Alive: %d\n", total);
}

int main(int argc, char* argv[])
{
  int iter;
  int* h_world;  //World on host
  int* tmpWorld; //tmp world pointer used to switch worlds

  FILE* fp;

  
  if (argc != 4) {
    fprintf(stderr, "usage: gameoflife <world dimension> <game steps> <output ppm file>\n");

    exit(1);
  }

  int dim = atoi(argv[1]); //Linear dimension of our world - without counting ghost cells
  int maxIter = atoi(argv[2]); //Number of game steps
  
  if ((fp = fopen(argv[3], "w")) == NULL) {
    fprintf(stderr, "can't create %s\n", argv[3]);
    exit(1);
  }

  size_t worldBytes = sizeof(int)*(dim+2)*(dim+2);

  // Allocate host World
  h_world =  malloc(worldBytes);
  
  //create initial world
  //initRandom(dim, h_world);
  initGlider(dim, h_world);
  
  int* h_newWorld = malloc(worldBytes);

  // --- Main loop ---
  for (iter = 0; iter<maxIter; iter++) {
    GOL_CPU(dim, h_world, h_newWorld);
    
    // Swap worlds
    tmpWorld = h_newWorld;
    h_newWorld = h_world;
    h_world = tmpWorld;
  }
    
  create_PPM(dim, h_world, fp);

  fclose(fp);

  return 0;
}
