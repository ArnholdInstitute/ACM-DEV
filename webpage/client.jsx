import React from "react";
import ReactDOM from "react-dom";
import {Router, Route, IndexRoute, hashHistory, IndexRedirect} from 'react-router';
import Landing from './pages/Landing'
import Layout from './Layout';
const app = document.getElementById('app');

ReactDOM.render(
	<Router history={hashHistory}>
		<Route path="/" component={Layout}>
			<IndexRedirect to="/landing"/>
			<Route path="landing" component={Landing}/>
		</Route>
	</Router>,
	app
);
