var debug = process.env.NODE_ENV !== "production";
var webpack = require('webpack');
var path = require('path');

module.exports = {
  context: path.resolve(__dirname + '/'),
  devtool: debug ? "inline-sourcemap" : null,
  entry: "./client.jsx",
  resolveLoader: {
    root: path.join(__dirname, 'node_modules')
  },
  resolve: {
    extensions: ['', '.js', '.jsx'],
    alias: {
      webworkify: 'webworkify-webpack',
    }
  },
  module: {
    loaders: [
      {
        test: /\.jsx?$/,
        exclude: /(node_modules|bower_components)/,
        loader: 'babel-loader',
        query: {
          presets: ['react', 'es2015', 'stage-0'],
          plugins: ['react-html-attrs', 'transform-class-properties', 'transform-decorators-legacy'],
        }
      },
      {
        test: /\.json$/,
        loader: 'json-loader'
      },
      { test: /\.css$/, loader: 'style!css' },
      {test : /\.less/, loader: 'css-loader!autoprefixer-loader!less-loader'},
      { test: /\.png$/, loader: "url-loader?limit=100000" },
      { test: /\.jpg$/, loader: "file-loader" }
    ],
    postLoaders: [{
      include: /node_modules\/mapbox-gl-shaders/,
      loader: 'transform',
      query: 'brfs'
    }]    
  },
  output: {
    path: __dirname + '/../',
    filename: 'client.min.js',
    publicPath: '/',
  },
  plugins: debug ? 
    [
    new webpack.optimize.OccurenceOrderPlugin(),
    new webpack.HotModuleReplacementPlugin(),
    new webpack.NoErrorsPlugin()
    ] : [
    new webpack.DefinePlugin({
      'process.env' : {'NODE_ENV' : JSON.stringify('production')}
    }),
    new webpack.optimize.DedupePlugin(),
    new webpack.optimize.OccurenceOrderPlugin(),
    new webpack.optimize.UglifyJsPlugin({ mangle: false, sourcemap: false }),
  ],
};