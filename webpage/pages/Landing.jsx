import React from 'react'
import {Row, Jumbotron} from 'react-bootstrap'

export default class Landing extends React.Component{

	render(){
		return(
			<div class="container">
				<Row>
					<Jumbotron>
					    <h1>Deep Learning Population Estimates</h1>
					    <p>
					    	A deep learning approach to estimating high resolution population density.
					    </p>
					</Jumbotron>
				</Row>
			</div>
		)
	}
}