from os import system
system("clear")
from flask import Flask, request, jsonify, render_template,make_response,redirect
from flask_restful import Api, Resource
from modules import evaluate,gitcheck,scrape

app = Flask(__name__)
api = Api(app)


# Variables to store the cumulative data
total_cost:float = 0.0
total_queries:int = 0

#Redirect to ragas Dashboard
class Home(Resource):
    def get(self):
        return redirect("Rag")

# Route to display the RAGAS dashboard in HTML and evaluate 
class Rag(Resource):
    def get(self):
        global total_cost, total_queries
        
        # Render the HTML template with total cost and queries
        return make_response(render_template('index.html', total_cost=total_cost, total_queries=total_queries))
    
    def post(self):
        global total_cost, total_queries

        data = request.get_json()

        # Extract cost and queries from the request
        cost = data.get('cost', 0.0)
        queries = data.get('queries', 0)

        # Update the total cost and queries
        total_cost += cost
        total_queries += queries

        # Print the data for verification (optional)
        print("Received Data:")
        # print(f"Questions: {data.get('question')}")
        # print(f"Contexts: {data.get('contexts')}")
        # print(f"Answer: {data.get('answer')}")
        # print(f"Ground Truth: {data.get('ground_truth')}")
        print(f"Cost: {cost}")
        print(f"Queries: {queries}")

        test_set = {"question":data.get('question'),"contexts":data.get('contexts'),"answer":data.get('answer'),"ground_truth":data.get('ground_truth')}

        result,df,image_path = evaluate.evaluate_rag(test_set)
        print("-----------------------------------")
        print(df)
        print(result)
        image_path = request.host_url+image_path
        print(f"Image Path: {image_path}")
        print("-----------------------------------")
        return jsonify({"message": result,"image_path":image_path})

# Route to display html form to take the repo url
class Gitanalyze(Resource):
    def get(self):
        # Render the HTML template with total cost and queries
        return make_response(render_template('form.html'))
    
    def post(self):
        data = request.get_json()
        system("rm -rf repos")
        report = gitcheck.main(REPO_URL=data['repoUrl'])
        return(report)
        # return make_response(render_template('form.html'))

#scrape data of stocks
class Scrape(Resource):
    def get(self):
        stock_name = request.args.get('name')
        
        if stock_name:
            # Placeholder processing logic
            result = scrape.main(stock_name)
            # Render the result on the index page
            return make_response(render_template('scrape.html', result=result))
        else:
            # If no name parameter is provided, return an empty result
            return make_response(render_template('scrape.html', result={}))

# Add the resource to the API
api.add_resource(Home, '/')
api.add_resource(Rag, '/rag')
api.add_resource(Gitanalyze, '/git')
api.add_resource(Scrape, '/scrape')

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",threaded=True,port=5000)
