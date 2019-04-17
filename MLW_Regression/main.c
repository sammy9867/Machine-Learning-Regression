#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#pragma warning(disable:4996)

#define MAXFLDS								200    /* maximum possible number of fields */
#define MAXFLDSIZE					         32    /* longest possible field + 1 = 31 byte field */
#define MULTI_LINEAR_ROWS			      10720    /* Number of rows in your Multi-Linear Regression CSV file*/ 
#define MULTI_LINEAR_NUMBER_OF_XS	       	  7    /* Number of columns in your Multi-Linear Regression CSV file that belong to X*/
#define BINARY_LOGISTIC_ROWS				673    /* Number of rows in your Binary Logistic Regression CSV file*/
#define BINARY_LOGISTIC_NUMBER_OF_XS	  	 11    /* Number of columns in your Binary Logistic Regression CSV file that belong to X*/
#define INCREMENT_SIZE						512    /* Increment size*/
#define THRESHOLD							0.5	   /* Threshold value for Binary Logistic Regression*/

size_t getline(char **, size_t *, FILE *);

void multi_linear_regression_solver(double*, double*, int, int, double, double, int);
void multi_linear_regression_train_helper(double**, double*, double*, double*, int, int);

void binary_logistic_regression_solver(double*, double*, int, int, double, double, int);
void binary_logistic_regression_train_helper(double**, double*, double*, double*, int, int);
void binary_logistic_regression_sigmoid(double *, double *);

double mean_squared_error(double*, double*, int);
double sum_of_squared_errors(double*, double*, int);
double mean_absolute_error(double*, double*, int);

int main() {

	//Choose a type of regression:
	int choose;
	printf("Select a type of regression :\n");
	printf("Please enter 1 for Multi-Linear Regression or 2 for Binary Logistic Regression:\n");
	scanf("%d",&choose);	
	if (choose == 1) {

        #pragma region Multi-Linear Regression
		printf("Multi-Linear Regression has been selected!\n");

		int numberOfXs;

		//Reading from CSV file
		char tmp[1024] = { 0x0 };
		int fldcnt = 0;
		char arr[MAXFLDS][MAXFLDSIZE] = { 0x0 };
		int recordcnt = 0;

		char *line = NULL;
		size_t len = 0;
		signed int read;
		unsigned long linenumber = 0;

		FILE *in;
		FILE *csvinputlog;
		char *csvinputLine = NULL;

		csvinputlog = fopen("..\\Input\\CSVInput.txt", "r");
	
		if (csvinputlog == NULL)
		{
			perror("File open error");
			exit(EXIT_FAILURE);
		}

		char input_path[10] = "..\\CSV\\";
		char input_csv[50];
		

		if (fgets(input_csv, 50, csvinputlog) != NULL) {
			/* writing content to stdout */
			puts(input_csv);
		}
			
	
		in = fopen(strcat(input_path, input_csv), "r");
		if (in == NULL)
		{
			perror("File open error");
			exit(EXIT_FAILURE);
		}
		

		double **Xs = NULL;
		double *Y = NULL;

		while ((read = getline(&line, &len, in)) != -1) {

			if (linenumber == 0){
				int numX = 0;
				for (int i = 0; i < read; i++)
				{
					if (line[i] == ';')
						numX++;  //Maximum number of Xs in the CSV.
				}
				numberOfXs = numX;
				
				Xs = (double**)malloc(numX * sizeof(double*));
				for (int i = 0; i < numX; i++)
				{
					Xs[i] = (double*)malloc(MULTI_LINEAR_ROWS * sizeof(double));
				}
				Y = (double*)malloc(MULTI_LINEAR_ROWS * sizeof(double));

				linenumber = linenumber + 1;
				continue;
			}
			for (int i = 0; i < read; i++)
			{
				if (line[i] == ',') line[i] = '.';
				if (line[i] == '\n' || line[i] == '\r') line[i] = 0;
			}
			//Replace the following number of X's with the total number of columns in your CSV - 1. The last column belongs to Y.
			sscanf_s(line, "%lf;%lf;%lf;%lf;%lf;%lf;%lf;%lf", &Xs[0][linenumber - 1], &Xs[1][linenumber - 1], &Xs[2][linenumber - 1], &Xs[3][linenumber - 1],
				&Xs[4][linenumber - 1], &Xs[5][linenumber - 1], &Xs[6][linenumber - 1], &Y[linenumber - 1]);
			linenumber = linenumber + 1;
		}
		printf("CSV has been read sucessfully!\n\n");

		double proportion, learning_rate;
		int iterations;


		printf("Enter the proportion to divide the dataset into train and test. Eg: 0.8 \n");
		scanf("%lf", &proportion);

		printf("Enter the number of iterations. Eg: 1000 \n");
		scanf("%d", &iterations);

		printf("Enter the learning rate. Eg: 0.01 \n");
		scanf("%lf", &learning_rate);
	
		printf("Number of X's : %d\n", numberOfXs);
		printf("Number of rows in CSV: %d\n", MULTI_LINEAR_ROWS);
		printf("Number of iterations: %d\n", iterations);
		printf("Learning rate: %lf\n", learning_rate);
		printf("Proportion: %lf\n",proportion);

		double _X[MULTI_LINEAR_NUMBER_OF_XS][MULTI_LINEAR_ROWS] = { 0x0 };
		for (int i = 0; i < MULTI_LINEAR_NUMBER_OF_XS; i++) {
			for (int j = 0; j < MULTI_LINEAR_ROWS; j++) {
				_X[i][j] = Xs[i][j];
			}
		}

		multi_linear_regression_solver((double *)_X, Y, MULTI_LINEAR_NUMBER_OF_XS, MULTI_LINEAR_ROWS, proportion, learning_rate, iterations);

		fclose(in);
		fclose(csvinputlog);

		for (int i = 0; i < MULTI_LINEAR_NUMBER_OF_XS; i++)
		{
			free(Xs[i]);
		}
		free(Xs);
		free(Y);

        #pragma endregion
		system("pause");

	}else if(choose == 2){
		printf("Binary Logistic Regression has been chosen\n");
        #pragma region Binary Logistic Regression
		int numberOfXs;

		//Reading from CSV file
		char tmp[1024] = { 0x0 };
		int fldcnt = 0;
		char arr[MAXFLDS][MAXFLDSIZE] = { 0x0 };
		int recordcnt = 0;

		char *line = NULL;
		size_t len = 0;
		signed int read;
		unsigned long linenumber = 0;
		char input_path[10] = "..\\CSV\\";
		char input_csv[50];


		char *csvinputLine = NULL;
		FILE *csvinputlog;

		csvinputlog = fopen("..\\Input\\CSVInput.txt", "r");

		if (csvinputlog == NULL)
		{
			perror("File open error");
			exit(EXIT_FAILURE);
		}

		if (fgets(input_csv, 50, csvinputlog) != NULL) {
			/* writing content to stdout */
			puts(input_csv);
		}

		FILE *in;
		//Replace the following path with a given CSV file.
		in = fopen(strcat(input_path, input_csv), "r");

		if (in == NULL)
		{
			perror("File open error");
			exit(EXIT_FAILURE);
		}

		double **Xs = NULL;
		double *Y = NULL;

		while ((read = getline(&line, &len, in)) != -1) {

			if (linenumber == 0) {
				int numX = 0;
				for (int i = 0; i < read; i++)
				{
					if (line[i] == ';')
						numX++;  //Maximum number of Xs in the CSV.
				}
				numberOfXs = numX;

				Xs = (double**)malloc(numX * sizeof(double*));
				for (int i = 0; i < numX; i++)
				{
					Xs[i] = (double*)malloc(BINARY_LOGISTIC_ROWS * sizeof(double));
				}
				Y = (double*)malloc(BINARY_LOGISTIC_ROWS * sizeof(double));

				linenumber = linenumber + 1;
				continue;
			}
			for (int i = 0; i < read; i++)
			{
				if (line[i] == ',') line[i] = '.';
				if (line[i] == '\n' || line[i] == '\r') line[i] = 0;
			}
			//Here, the first column is Y and the following belong to X.
			sscanf_s(line, "%lf;%lf;%lf;%lf;%lf;%lf;%lf;%lf;%lf;%lf;%lf;%lf", &Y[linenumber - 1], &Xs[0][linenumber - 1], &Xs[1][linenumber - 1], &Xs[2][linenumber - 1], &Xs[3][linenumber - 1],
				&Xs[4][linenumber - 1], &Xs[5][linenumber - 1], &Xs[6][linenumber - 1], &Xs[7][linenumber - 1], &Xs[8][linenumber - 1], &Xs[9][linenumber - 1], &Xs[10][linenumber - 1]);

			linenumber = linenumber + 1;
		}
		printf("CSV has been read sucessfully!\n\n");


		double proportion, learning_rate;
		int iterations;

		printf("Enter the proportion to divide the dataset into train and test. Eg: 0.8 \n");
		scanf("%lf", &proportion);

		printf("Enter the number of iterations. Eg: 1000 \n");
		scanf("%d", &iterations);

		printf("Enter the learning rate. Eg: 0.01 \n");
		scanf("%lf", &learning_rate);


		printf("Number of X's : %d\n", BINARY_LOGISTIC_NUMBER_OF_XS);
		printf("Number of rows in CSV: %d\n", BINARY_LOGISTIC_ROWS);
		printf("Number of iterations: %d\n", iterations);
		printf("Learning rate: %lf\n", learning_rate);
		printf("Proportion: %lf\n", proportion);


		double _X[BINARY_LOGISTIC_NUMBER_OF_XS][BINARY_LOGISTIC_ROWS] = { 0x0 };
		for (int i = 0; i < BINARY_LOGISTIC_NUMBER_OF_XS; i++) {
			for (int j = 0; j < BINARY_LOGISTIC_ROWS; j++) {
				_X[i][j] = Xs[i][j];
			}
		}

		binary_logistic_regression_solver((double *)_X, Y, BINARY_LOGISTIC_NUMBER_OF_XS, BINARY_LOGISTIC_ROWS, proportion, learning_rate, iterations);

		fclose(in);
		fclose(csvinputlog);

		for (int i = 0; i < BINARY_LOGISTIC_NUMBER_OF_XS; i++)
		{
			free(Xs[i]);
		}
		free(Xs);
		free(Y);
        #pragma endregion
		system("pause");
	}
	else {
		printf("Try again later!");
		return 0;
	}

}

size_t getline(char **lineptr, size_t *n, FILE *stream) {
	char *bufptr = NULL;
	char *p = bufptr;
	size_t size;
	int c;

	if (lineptr == NULL) {
		return -1;
	}
	if (stream == NULL) {
		return -1;
	}
	if (n == NULL) {
		return -1;
	}
	bufptr = *lineptr;
	size = *n;

	c = fgetc(stream);
	if (c == EOF) {
		return -1;
	}
	if (bufptr == NULL) {
		bufptr = malloc(128);
		if (bufptr == NULL) {
			return -1;
		}
		size = 128;
	}
	p = bufptr;
	while (c != EOF) {
		if ((p - bufptr) > (size - 1)) {
			size = size + 128;
			bufptr = realloc(bufptr, size);
			if (bufptr == NULL) {
				return -1;
			}
		}
		*p++ = c;
		if (c == '\n') {
			break;
		}
		c = fgetc(stream);
	}

	*p++ = '\0';
	*lineptr = bufptr;
	*n = size;

	return p - bufptr - 1;
}

void multi_linear_regression_solver(double* _X, double* _Y, int nX, int rows, double proportion, double learning_rate, int iterations) {

    #pragma region IntializingArrays

	FILE *output;
	if (output = fopen("..\\OutputLog\\multi_linear_regression_output.txt", "r")){
		remove("..\\OutputLog\\multi_linear_regression_output.txt");
	     	output = fopen("..\\OutputLog\\multi_linear_regression_output.txt", "w");
			if (output == NULL)
			{
				printf("Error opening file!\n");
					exit(1);
			}
	}
	else {
		output = fopen("..\\OutputLog\\multi_linear_regression_output.txt", "w");
			if (output == NULL)
			{
				printf("Error opening file!\n");
					exit(1);
			}
	}
	
	
	//Size of training and test data sets:
	int trainingSize = (int)ceil(rows*proportion);
	int testingSize = rows - trainingSize;
	fprintf(output, "Total number of rows: %d has been split into %d for training and %d for testing.\n", rows, trainingSize, testingSize);


	//Training Arrays
	double** trainingX = (double**)malloc(sizeof(double*) * nX);
	double*  trainingY = malloc(sizeof(double) * trainingSize);

	//Testing Arrays
	double** testX = (double**)malloc(sizeof(double*) * nX);
	double*  testY = malloc(sizeof(double) * testingSize);

	//Normalized Training and Testing Arrays
	double** NtrainingX = (double**)malloc(sizeof(double*) * nX);
	double* NtrainingY = malloc(sizeof(double) * trainingSize);

	double** NtestX = (double**)malloc(sizeof(double*) * nX);
	double* NtestingY = malloc(sizeof(double) * testingSize);

	//Predicted Y
	double* predictedY = malloc(sizeof(double) * testingSize);


	//Intializing 2d arrays
	for (int i = 0; i < nX; i++)
	{
		trainingX[i] = (double *)malloc(sizeof(double) * trainingSize);
		NtrainingX[i] = (double*)malloc(sizeof(double) * trainingSize);
		testX[i] = (double*)malloc(sizeof(double) * testingSize);
		NtestX[i] = (double*)malloc(sizeof(double) * testingSize);
	}

	//Min and Max of Training and Testing arrays used during normalization.
	double* minTrainXs = (double*)malloc(sizeof(double)*nX);
	double* maxTrainXs = (double*)malloc(sizeof(double)*nX);
	double* minTestXs = (double*)malloc(sizeof(double)*nX);
	double* maxTestXs = (double*)malloc(sizeof(double)*nX);

	double mintestY = 0.0;
	double maxtestY = 0.0;

	double  mintrainY = 0.0;
	double  maxtrainY = 0.0;
    #pragma endregion

    #pragma region SplittingIntoTrainAndTest

	//TrainingX
	fprintf(output, "\n ******************************* \n");
	for (int nx = 0; nx < nX; nx++) {
		for (int i = 0; i < trainingSize; i++) {
			trainingX[nx][i] = *((_X + nx * rows) + i);
			fprintf(output,"TrainingX: [%d][%d] %lf\n", nx,i, trainingX[nx][i]);
			if (0 == i)
			{
				minTrainXs[nx] = trainingX[nx][i];
				maxTrainXs[nx] = trainingX[nx][i];
			}
			else
			{
				minTrainXs[nx] = (minTrainXs[nx] < trainingX[nx][i]) ? minTrainXs[nx] : trainingX[nx][i];
				maxTrainXs[nx] = (maxTrainXs[nx] > trainingX[nx][i]) ? maxTrainXs[nx] : trainingX[nx][i];
			}

		}
	}

	//TestingX
	fprintf(output, "\n ******************************* \n");
	for (int nx = 0; nx < nX; nx++) {
		for (int i = 0; i < testingSize; i++) {
			testX[nx][i] = *((_X + nx * rows) + (i + trainingSize)); 
			fprintf(output, "TestingX: [%d][%d] %lf\n", nx, i, testX[nx][i]);
			if (0 == i)
			{
				minTestXs[nx] = testX[nx][i];
				maxTestXs[nx] = testX[nx][i];
			}
			else
			{
				minTestXs[nx] = (minTestXs[nx] < testX[nx][i]) ? minTestXs[nx] : testX[nx][i];
				maxTestXs[nx] = (maxTestXs[nx] > testX[nx][i]) ? maxTestXs[nx] : testX[nx][i];
			}
		}
	}

	//TrainingY
	fprintf(output, "\n ******************************* \n");
	for (int i = 0; i < trainingSize; i++) {
		trainingY[i] = _Y[i];
		fprintf(output, "TrainingY: [%d] %lf\n", i, trainingY[i]);
		if (0 == i)
		{
			mintrainY = trainingY[i];
			maxtrainY = trainingY[i];
		}
		else
		{
			mintrainY = mintrainY < trainingY[i] ? mintrainY : trainingY[i];
			maxtrainY = maxtrainY > trainingY[i] ? maxtrainY : trainingY[i];
		}
	}

	//Testing Y
	fprintf(output, "\n ******************************* \n");
	for (int j = trainingSize; j < rows; j++) {
		testY[j - trainingSize] = _Y[j];
		fprintf(output, "TestingY: [%d] %lf\n", j - trainingSize, testY[j - trainingSize]);
		if (0 == j)
		{
			mintestY = testY[j - trainingSize];
			maxtestY = testY[j - trainingSize];
		}
		else
		{
			mintestY = mintestY < testY[j - trainingSize] ? mintestY : testY[j - trainingSize];
			maxtestY = maxtestY > testY[j - trainingSize] ? maxtestY : testY[j - trainingSize];
		}
	}
	

    #pragma endregion

    #pragma region Normalization

	//Training
	fprintf(output, "\n ******************************* \n");
	for (int nx = 0; nx < nX; nx++){
		for (int i = 0; i < trainingSize; i++) {
			NtrainingX[nx][i] = (double)(trainingX[nx][i] - minTrainXs[nx]) / (double)(maxTrainXs[nx] - minTrainXs[nx]);
			fprintf(output, "Normalized trainingX: [%d][%d] %lf\n", nx, i, NtrainingX[nx][i]);
			if (0 == nx)
			{
				NtrainingY[i] = (trainingY[i] - mintrainY) / (maxtrainY - mintrainY);
			}
		}
	}

	//Testing
	fprintf(output, "\n ******************************* \n");
	for (int nx = 0; nx < nX; nx++){
		for (int i = 0; i < testingSize; i++) {
			NtestX[nx][i] = (testX[nx][i] - minTestXs[nx]) / (maxTestXs[nx] - minTestXs[nx]);
			fprintf(output, "Normalized testingX: [%d][%d] %lf\n", nx, i, NtestX[nx][i]);
			if (0 == nx)
			{
				NtestingY[i] = (testY[i] - mintestY) / (maxtestY - mintestY);
			}
		}
	}

	//Printing TrainingY
	fprintf(output, "\n ******************************* \n");
	for (int i = 0; i < trainingSize; i++) {
		fprintf(output, "Normalized trainingY: [%d] %lf\n",i, NtrainingY[i]);
	}

	//Printing TestingY
	fprintf(output, "\n ******************************* \n");
	for (int i = 0; i < testingSize; i++) {
		fprintf(output, "Normalized testingY: [%d] %lf\n",i, NtestingY[i]);
	}

    #pragma endregion
	 
    #pragma region GradientDescent

	//Training
	double *theta_training = malloc(sizeof(double) * (nX + 1));
	memset(theta_training, 0, sizeof(double) * (nX + 1));

	double *theta_testing = malloc(sizeof(double) * (nX + 1));
	memset(theta_testing, 0, sizeof(double) * (nX + 1));

	double *training_helper = malloc(sizeof(double) * (nX + 1));
	memset(training_helper, 0, sizeof(double) * (nX + 1));

	fprintf(output, "\n ******************************* \n"); 
	fprintf(output, "\n GRADIENT DESCENT \n");
	fprintf(output, "Theta values for training set: \n");
	for (int i = 0; i < iterations; i++) {
		
		multi_linear_regression_train_helper(NtrainingX, NtrainingY, theta_training, training_helper, trainingSize, (nX+1));
		for (int col = 0; col < (nX + 1); col++) {

			theta_training[col] = theta_training[col] - (double)(learning_rate / rows) * training_helper[col];
			fprintf(output,"%lf ", theta_training[col]);
		}
		fprintf(output, "\n");
	}

	printf("\nTheta values:\n");
	for (int i = 0; i < (nX + 1); i++) {
		printf("%lf ", theta_training[i]);
	}

	
	fprintf(output,"\n\n----------------------------------\n\n");
	fprintf(output,"Predictions of Y: \n");

	printf("Predictions of Y: \n");
	for (int i = 0; i < testingSize; i++) {
		predictedY[i] = theta_training[0];
		for (int j = 0; j < nX; j++)
			predictedY[i] += theta_training[j + 1] * NtestX[j][i];
		
		printf("%PredictedY: [%d] %lf \n", i, predictedY[i]);
		fprintf(output,"%PredictedY: [%d] %lf\n", i, predictedY[i]);
	}

	double MSE = mean_squared_error(NtestingY, predictedY, testingSize);
	double SSE = sum_of_squared_errors(NtestingY, predictedY, testingSize);
	double MAE = mean_absolute_error(NtestingY, predictedY, testingSize);
	
	//Calculate formulas for testing
	fprintf(output, "\n\n----------------------------------\n\n");
	fprintf(output, "TestingSet: Mean Squared Error:%lf\n", MSE);
	fprintf(output, "TestingSet: Sum of Squared Errors:%lf\n", SSE);
	fprintf(output, "TestingSet: Mean Absolute Error: %lf\n", MAE);

	printf("\n\n");
	printf("TestingSet: Mean Squared Error: %lf\n",MSE);
	printf("TestingSet: Sum of Squared Errors: %lf\n",SSE);
	printf("TestingSet: Mean Absolute Error: %lf\n",MAE);
    #pragma endregion
	
    #pragma region DeallocatingArrays
	fclose(output);
	for (int i = 0; i < nX; i++)
	{
		free(trainingX[i]);
		free(NtrainingX[i]);
		free(testX[i]);
		free(NtestX[i]);
	}

	free(trainingX);
	free(NtrainingX);
	free(testX);
	free(NtestX);
	free(minTrainXs);
	free(minTestXs);
	free(maxTrainXs);
	free(maxTestXs);
	free(theta_training);
	free(theta_testing);
	free(training_helper);
	free(predictedY);
    #pragma endregion
}

void multi_linear_regression_train_helper(double** NtrainX, double* NtrainY, double* theta, double* training_helper, int rows, int cols) {

	double *step1 = malloc(sizeof(double) * rows);
	memset(step1, 0, sizeof(double) * rows);

	double value = 0.0;

	double **setOfNorXs = malloc(rows * sizeof(double*));
	double **step2 = malloc(rows * sizeof(double*));

	for (int i = 0; i < rows; ++i) {
		setOfNorXs[i] = malloc(cols * sizeof(double));
		step2[i] = malloc(cols * sizeof(double));
	}


	for (int i = 0; i < rows; i++) {
		setOfNorXs[i][0] = 1.0;
	}

	for (int i = 1; i < cols; i++) {
		for (int nx = 0; nx < rows; nx++) {
			setOfNorXs[nx][i] = NtrainX[i - 1][nx];
		}
	}

	for (int nx = 0; nx < rows; nx++) {
		for (int i = 0; i < cols; i++) {
			step2[nx][i] = setOfNorXs[nx][i];
		}
	}


	//Step 1: Matrix multiplication of (X * theta) - Y 
	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {

			value += setOfNorXs[row][col] * theta[col];
			if (col == (cols-1)) {
				step1[row] = value - NtrainY[row];
				value = 0.0;
			}
		}
	}

	//Step 2: X * (Matrix multiplication of (X * theta) - Y )
	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {
			step2[row][col] *= step1[row];
		}
	}


	//Step 3: Sum of [X * (Matrix multiplication of (X * theta) - Y )] columnwise
	double sum = 0.0;
	for (int col = 0; col < cols; col++) {
		for (int row = 0; row < rows; row++) {
			sum += step2[row][col];
			training_helper[col] = sum;
		}
		sum = 0.0;
	}

	//Deallocation
	free(step1);
	for (int i = 0; i < rows; i++) {
		free(setOfNorXs[i]);
		free(step2[i]);
	}
	free(setOfNorXs);
	free(step2);
}

void binary_logistic_regression_solver(double* _X, double* _Y, int nX, int rows, double proportion, double learning_rate, int iterations) {

#pragma region IntializingArrays

	FILE *output;
	if (output = fopen("..\\OutputLog\\binary_logistic_regression_output.txt", "r")) {
		remove("..\\OutputLog\\binary_logistic_regression_output.txt");
		output = fopen("..\\OutputLog\\binary_logistic_regression_output.txt", "w");
		if (output == NULL)
		{
			printf("Error opening file!\n");
			exit(1);
		}
	}
	else {
		output = fopen("..\\OutputLog\\binary_logistic_regression_output.txt", "w");
		if (output == NULL)
		{
			printf("Error opening file!\n");
			exit(1);
		}
	}


	//Size of training and test data sets:
	int trainingSize = (int)ceil(rows*proportion);
	int testingSize = rows - trainingSize;
	fprintf(output, "Total number of rows: %d has been split into %d for training and %d for testing.\n", rows, trainingSize, testingSize);


	//Training Arrays
	double** trainingX = (double**)malloc(sizeof(double*) * nX);
	double*  trainingY = malloc(sizeof(double) * trainingSize);

	//Testing Arrays
	double** testX = (double**)malloc(sizeof(double*) * nX);
	double*  testY = malloc(sizeof(double) * testingSize);

	//Normalized Training and Testing Arrays
	double** NtrainingX = (double**)malloc(sizeof(double*) * nX);
	double* NtrainingY = malloc(sizeof(double) * trainingSize);

	double** NtestX = (double**)malloc(sizeof(double*) * nX);
	double* NtestingY = malloc(sizeof(double) * testingSize);

	//Predicted Y
	double* predictedY = malloc(sizeof(double) * testingSize);


	//Intializing 2d arrays
	for (int i = 0; i < nX; i++)
	{
		trainingX[i] = (double *)malloc(sizeof(double) * trainingSize);
		NtrainingX[i] = (double*)malloc(sizeof(double) * trainingSize);
		testX[i] = (double*)malloc(sizeof(double) * testingSize);
		NtestX[i] = (double*)malloc(sizeof(double) * testingSize);
	}

	//Min and Max of Training and Testing arrays used during normalization.
	double* minTrainXs = (double*)malloc(sizeof(double)*nX);
	double* maxTrainXs = (double*)malloc(sizeof(double)*nX);
	double* minTestXs = (double*)malloc(sizeof(double)*nX);
	double* maxTestXs = (double*)malloc(sizeof(double)*nX);

	double mintestY = 0.0;
	double maxtestY = 0.0;

	double  mintrainY = 0.0;
	double  maxtrainY = 0.0;
#pragma endregion

#pragma region SplittingIntoTrainAndTest

	//TrainingX
	fprintf(output, "\n ******************************* \n");
	for (int nx = 0; nx < nX; nx++) {
		for (int i = 0; i < trainingSize; i++) {
			trainingX[nx][i] = *((_X + nx * rows) + i);
			fprintf(output, "TrainingX: [%d][%d] %lf\n", nx, i, trainingX[nx][i]);
			if (0 == i)
			{
				minTrainXs[nx] = trainingX[nx][i];
				maxTrainXs[nx] = trainingX[nx][i];
			}
			else
			{
				minTrainXs[nx] = (minTrainXs[nx] < trainingX[nx][i]) ? minTrainXs[nx] : trainingX[nx][i];
				maxTrainXs[nx] = (maxTrainXs[nx] > trainingX[nx][i]) ? maxTrainXs[nx] : trainingX[nx][i];
			}

		}
	}

	//TestingX
	fprintf(output, "\n ******************************* \n");
	for (int nx = 0; nx < nX; nx++) {
		for (int i = 0; i < testingSize; i++) {
			testX[nx][i] = *((_X + nx * rows) + (i + trainingSize));
			fprintf(output, "TestingX: [%d][%d] %lf\n", nx, i, testX[nx][i]);
			if (0 == i)
			{
				minTestXs[nx] = testX[nx][i];
				maxTestXs[nx] = testX[nx][i];
			}
			else
			{
				minTestXs[nx] = (minTestXs[nx] < testX[nx][i]) ? minTestXs[nx] : testX[nx][i];
				maxTestXs[nx] = (maxTestXs[nx] > testX[nx][i]) ? maxTestXs[nx] : testX[nx][i];
			}
		}
	}

	//TrainingY
	fprintf(output, "\n ******************************* \n");
	for (int i = 0; i < trainingSize; i++) {
		trainingY[i] = _Y[i];
		fprintf(output, "TrainingY: [%d] %lf\n", i, trainingY[i]);
		if (0 == i)
		{
			mintrainY = trainingY[i];
			maxtrainY = trainingY[i];
		}
		else
		{
			mintrainY = mintrainY < trainingY[i] ? mintrainY : trainingY[i];
			maxtrainY = maxtrainY > trainingY[i] ? maxtrainY : trainingY[i];
		}
	}

	//Testing Y
	fprintf(output, "\n ******************************* \n");
	for (int j = trainingSize; j < rows; j++) {
		testY[j - trainingSize] = _Y[j];
		fprintf(output, "TestingY: [%d] %lf\n", j - trainingSize, testY[j - trainingSize]);
		if (0 == j)
		{
			mintestY = testY[j - trainingSize];
			maxtestY = testY[j - trainingSize];
		}
		else
		{
			mintestY = mintestY < testY[j - trainingSize] ? mintestY : testY[j - trainingSize];
			maxtestY = maxtestY > testY[j - trainingSize] ? maxtestY : testY[j - trainingSize];
		}
	}


#pragma endregion

#pragma region Normalization

	//Training
	fprintf(output, "\n ******************************* \n");
	for (int nx = 0; nx < nX; nx++) {
		for (int i = 0; i < trainingSize; i++) {
			NtrainingX[nx][i] = (double)(trainingX[nx][i] - minTrainXs[nx]) / (double)(maxTrainXs[nx] - minTrainXs[nx]);
			fprintf(output, "Normalized trainingX: [%d][%d] %lf\n", nx, i, NtrainingX[nx][i]);
			if (0 == nx)
			{
				NtrainingY[i] = (trainingY[i] - mintrainY) / (maxtrainY - mintrainY);
			}
		}
	}

	//Testing
	fprintf(output, "\n ******************************* \n");
	for (int nx = 0; nx < nX; nx++) {
		for (int i = 0; i < testingSize; i++) {
			NtestX[nx][i] = (testX[nx][i] - minTestXs[nx]) / (maxTestXs[nx] - minTestXs[nx]);
			fprintf(output, "Normalized testingX: [%d][%d] %lf\n", nx, i, NtestX[nx][i]);
			if (0 == nx)
			{
				NtestingY[i] = (testY[i] - mintestY) / (maxtestY - mintestY);
			}
		}
	}

	//Printing TrainingY
	fprintf(output, "\n ******************************* \n");
	for (int i = 0; i < trainingSize; i++) {
		fprintf(output, "Normalized trainingY: [%d] %lf\n", i, NtrainingY[i]);
	}

	//Printing TestingY
	fprintf(output, "\n ******************************* \n");
	for (int i = 0; i < testingSize; i++) {
		fprintf(output, "Normalized testingY: [%d] %lf\n", i, NtestingY[i]);
	}

#pragma endregion

#pragma region GradientDescent

	//Training
	double *theta_training = malloc(sizeof(double) * (nX + 1));
	memset(theta_training, 0, sizeof(double) * (nX + 1));

	double *theta_testing = malloc(sizeof(double) * (nX + 1));
	memset(theta_testing, 0, sizeof(double) * (nX + 1));

	double *training_helper = malloc(sizeof(double) * (nX + 1));
	memset(training_helper, 0, sizeof(double) * (nX + 1));

	fprintf(output, "\n ******************************* \n");
	fprintf(output, "\n GRADIENT DESCENT \n");
	fprintf(output, "Theta values for training set: \n");
	for (int i = 0; i < iterations; i++) {

		binary_logistic_regression_train_helper(NtrainingX, NtrainingY, theta_training, training_helper, trainingSize, (nX + 1));
		for (int col = 0; col < (nX + 1); col++) {

			theta_training[col] = theta_training[col] - (double)(learning_rate / rows) * training_helper[col];
			fprintf(output, "%lf ", theta_training[col]);
		}
		fprintf(output, "\n");
	}

	printf("\nTheta values:\n");
	for (int i = 0; i < (nX + 1); i++) {
		printf("%lf ", theta_training[i]);
	}

	fprintf(output, "\n\n----------------------------------\n\n");
	fprintf(output, "Predictions of Y: \n");

	double *sigmoidArr = malloc(sizeof(double) * (testingSize));
	double *classifaction = malloc(sizeof(double) * (testingSize));

	printf("Predictions of Y: \n");
	for (int i = 0; i < testingSize; i++) {
		predictedY[i] = theta_training[0];
		 for (int j = 0; j < nX; j++)
			predictedY[i] += theta_training[j + 1] * NtestX[j][i];
		    binary_logistic_regression_sigmoid(predictedY + i, sigmoidArr + i);
			printf("PredictedY: [%d] %lf\n", i, predictedY[i]);
			fprintf(output, "PredictedY: [%d] %lf\n", i, predictedY[i]);
	}

	fprintf(output, "\n\n----------------------------------\n\n");
	for (int i = 0; i < testingSize; i++) {
		classifaction[i] = sigmoidArr[i] > THRESHOLD ? 1.0 : 0.0;
		printf("[%d] Probabilties: %lf and it's Classification: %0.f\n", i, sigmoidArr[i],classifaction[i]);
		fprintf(output, "[%d] Probabilties: %lf and it's Classification: %0.f\n", i, sigmoidArr[i], classifaction[i]);
	}



	double MSE = mean_squared_error(NtestingY, predictedY, testingSize);
	double SSE = sum_of_squared_errors(NtestingY, predictedY, testingSize);
	double MAE = mean_absolute_error(NtestingY, predictedY, testingSize);

	//Calculate formulas for testing
	fprintf(output, "\n\n----------------------------------\n\n");
	fprintf(output, "TestingSet: Mean Squared Error:%lf\n", MSE);
	fprintf(output, "TestingSet: Sum of Squared Errors:%lf\n", SSE);
	fprintf(output, "TestingSet: Mean Absolute Error: %lf\n", MAE);

	printf("\n\n");
	printf("TestingSet: Mean Squared Error: %lf\n", MSE);
	printf("TestingSet: Sum of Squared Errors: %lf\n", SSE);
	printf("TestingSet: Mean Absolute Error: %lf\n", MAE);
#pragma endregion

#pragma region DeallocatingArrays
	fclose(output);
	for (int i = 0; i < nX; i++)
	{
		free(trainingX[i]);
		free(NtrainingX[i]);
		free(testX[i]);
		free(NtestX[i]);
	}

	free(trainingX);
	free(NtrainingX);
	free(testX);
	free(NtestX);
	free(minTrainXs);
	free(minTestXs);
	free(maxTrainXs);
	free(maxTestXs);
	free(theta_training);
	free(theta_testing);
	free(training_helper);
	free(predictedY);
	free(sigmoidArr);
	free(classifaction);
#pragma endregion
}

void binary_logistic_regression_train_helper(double** NtrainX, double* NtrainY, double* theta, double* training_helper, int rows, int cols) {

	double *step1 = malloc(sizeof(double) * rows);
	memset(step1, 0, sizeof(double) * rows);

	double value = 0.0;

	double **setOfNorXs = malloc(rows * sizeof(double*));
	double **step2 = malloc(rows * sizeof(double*));

	for (int i = 0; i < rows; ++i) {
		setOfNorXs[i] = malloc(cols * sizeof(double));
		step2[i] = malloc(cols * sizeof(double));
	}


	for (int i = 0; i < rows; i++) {
		setOfNorXs[i][0] = 1.0;
	}

	for (int i = 1; i < cols; i++) {
		for (int nx = 0; nx < rows; nx++) {
			setOfNorXs[nx][i] = NtrainX[i - 1][nx];
		}
	}

	for (int nx = 0; nx < rows; nx++) {
		for (int i = 0; i < cols; i++) {
			step2[nx][i] = setOfNorXs[nx][i];
		}
	}


	//Step 1: Matrix multiplication of (X * theta) - Y 
	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {

			value += setOfNorXs[row][col] * theta[col];
			if (col == (cols - 1)) {
				step1[row] = value - NtrainY[row];
				value = 0.0;
			}
		}
	}

	//Step 2: X * (Matrix multiplication of (X * theta) - Y )
	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {
			step2[row][col] *= step1[row];
		}
	}


	//Step 3: Sum of [X * (Matrix multiplication of (X * theta) - Y )] columnwise
	double sum = 0.0;
	for (int col = 0; col < cols; col++) {
		for (int row = 0; row < rows; row++) {
			sum += step2[row][col];
			training_helper[col] = sum;
		}
		sum = 0.0;
	}

	//Deallocation
	free(step1);
	for (int i = 0; i < rows; i++) {
		free(setOfNorXs[i]);
		free(step2[i]);
	}
	free(setOfNorXs);
	free(step2);
}

void binary_logistic_regression_sigmoid(double* predictedY, double* result) {

	*result = 1.0 / (1 + exp(-(*predictedY)));

	return;
}

double mean_squared_error(double* observedY, double* predictedY, int size) {

	double mse_sum = 0.0;
	for (int i = 0; i < size; i++) {
		mse_sum += pow((observedY[i] - predictedY[i]),2);
	}
	double MSE = mse_sum / (double)size;
	return MSE;
}

double sum_of_squared_errors(double* observedY, double* predictedY, int size) {

	double SSE = 0.0;
	for (int i = 0; i < size; i++) {
		SSE += pow((observedY[i] - predictedY[i]), 2);
	}
	return SSE;
}

double mean_absolute_error(double* observedY, double* predictedY, int size) {

	double mae_sum = 0.0;
	for (int i = 0; i < size; i++) {
		mae_sum += fabs(observedY[i] - predictedY[i]);
	}
	double MAE = mae_sum / (double)size;
	return MAE;
}