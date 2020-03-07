
* You run the application like so: 'python hello_flask.py'. This will start your app on localhost, port 5000.

* The @app.route('/') decorator ties the root URL to the index() function. Therefore, when a user goes the root URL: http://localhost:5000/, the index() function is automatically invoked.

* request.args.get("name") retrieves the name parameter from URL. For example, if you specify a URL like so: http://localhost:5000/?name=Bob, the name parameter will contain the value “Bob”.

* render_template("index.html", name=name) sends your data to a template and returns the rendered HTML to the browser. More on templating in a minute.

* Lastly, app.run(port=5000, debug=True) runs your Flask application on the designated port, and crucially sets debug=True. With debug enabled, Flask will automatically check for code changes and auto-reload these changes. No need to kill Flask and restart it each time you make code changes!