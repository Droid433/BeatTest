
 
var spawn = require('child_process').spawn,
    py    = spawn('python', ['tsp_google_tools.py']),
   other = '', 
    data =  [[-32.26976149157331,
      -32.45151641822355
    ],
             [125.4854250121665,
              123.97588494072352,
            ]]
      ,
      
    dataString = '';

py.stdout.on('data', function(data){
  dataString+=data;
});
py.stderr.on('data', function(data){
  dataString+=data;
})
py.stdout.on('end', function(){
  console.log(dataString);
  other = dataString;
});

py.stdin.write(JSON.stringify(data));
py.stdin.end();
