functions {
    matrix selfmade_relu(matrix z, int a_rows, int a_columns){
      matrix[a_rows, a_columns] a;
      for (i in 1:a_rows){
        for (j in 1:a_columns){
          if(z[i,j]>0){
            a[i,j]=z[i,j];
          }
          else{
            a[i,j]=0;
          }
        }
      }
      return a;
    }
    
    vector calculate_mu(matrix X, matrix bias_first_m, real bias_output, matrix w_first, vector w_output, int num_layers) {
		int N = rows(X);
		int num_nodes = rows(w_first);

		matrix[N, num_nodes] layer_values[num_layers - 2];
		vector[N] mu;

		//layer_values[1] = selfmade_relu(bias_first_m + X * w_first',N,num_nodes);  
    layer_values[1] = inv_logit(bias_first_m + X * w_first');   

		mu = bias_output + layer_values[num_layers - 2] * w_output;

      return mu;
    }
  }
  data {
    int<lower=0> N;						// num data
    int<lower=0> d;						// dim x
    int<lower=0> num_nodes;				// num hidden unites
    int<lower=1> num_middle_layers;		// num hidden layer
    matrix[N,d] X;						// X
    real y[N];							// y
	int<lower=0> Nt;					// num predicive data
	matrix[Nt,d] Xt;					// X predicive
	real<lower=0> sigma;				// const sigma
  }
  transformed data {
    int num_layers;
    num_layers = num_middle_layers + 2;
  }
  parameters {
    vector[num_nodes] bias_first;
    real bias_output;
    matrix[num_nodes, d] w_first;
    vector[num_nodes] w_output;
	// hyperparameters
    real<lower=0> bias_first_h;
    real<lower=0> w_first_h;
    real<lower=0> w_output_h;
  } 
  transformed parameters {
    matrix[N, num_nodes] bias_first_m = rep_matrix(bias_first', N);
  }
  model{
    vector[N] mu;
    mu = calculate_mu(X, bias_first_m, bias_output, w_first, w_output, num_layers);
    y ~ normal(mu,sigma);
    
    //priors
    bias_first_h ~ normal(0, 1);
    bias_first ~ normal(0, 1);
    bias_output ~ normal(0, 1);

    w_first_h ~ normal(0, 1);
    to_vector(w_first) ~ normal(0, 1);

    w_output_h ~ normal(0, 1);
    w_output ~ normal(0, 1);
  }
  generated quantities{
    vector[Nt] predictions;
	matrix[Nt, num_nodes] bias_first_mg = rep_matrix(bias_first', Nt);
	vector[Nt] mu;

	mu = calculate_mu(Xt, bias_first_mg, bias_output,w_first, w_output, num_layers);
	for(i in 1:Nt){ 
		predictions[i] = normal_rng(mu[i],sigma);
	}
}


  