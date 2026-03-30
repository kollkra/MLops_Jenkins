pipeline {
    agent any
    
    environment {
        BUILD_ID = 'dontKillMe'
        JENKINS_NODE_COOKIE = 'dontKillMe'
    }

    stages {
        stage('Download') {
            steps {
                script {
                    sh '''
                        cd ${WORKSPACE}
                        python3 -m venv ./my_env
                        . ./my_env/bin/activate
                        pip3 install --upgrade pip
                        pip3 install setuptools
                        pip3 install -r requirements.txt
                        python3 download.py
                    '''
                }
            }
        }
        
        stage('Train') {
            steps {
                script {
                    sh '''
                        cd ${WORKSPACE}
                        . ./my_env/bin/activate
                        python3 train_model.py
                    '''
                }
            }
        }
        
        stage('Deploy') {
            steps {
                script {
                    sh '''
                        cd ${WORKSPACE}
                        . ./my_env/bin/activate
                        path_model=$(cat best_model.txt)
                        nohup mlflow models serve -m $path_model -p 5003 --no-conda > mlflow.log 2>&1 &
                        sleep 15
                    '''
                }
            }
        }
        
        stage('Status') {
            steps {
                script {
                    sh '''
                        sleep 5
                        curl -v http://127.0.0.1:5003/invocations \
                            -H "Content-Type: application/json" \
                            --data '{"inputs": [[5.0, 1.0, 2.0, 1.0, 3.0, 50000.0, 2000.0, 15000.0]]}'
                    '''
                }
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'best_model.txt, lr_cars.pkl', allowEmptyArchive: true
        }
    }
}
