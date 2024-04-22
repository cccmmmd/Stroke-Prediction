d3.csv('https://raw.githubusercontent.com/cccmmmd/Stroke-Prediction/main/healthcare-dataset-stroke-data.csv').then(
    res => {
        // console.log(res)
        drawMarried(res)
        drawSmoke(res)
    }
);



function drawMarried(res) {
    let myGraph = document.getElementById('myGraph1');
    let trace1 ={};
    trace1.type ="bar";
    trace1.name ="中風";
    trace1.x =[];
    trace1.y =[];

    count_y_s = 0;
    count_y_n = 0;

    count_n_s = 0;
    count_n_n = 0;
    res.forEach((e) => {
        if(e.ever_married == 'Yes'){
            if (e.stroke == "1"  ){
                count_y_s += 1;
            }else{
                count_y_n += 1;
            }
        }else{
            if (e.stroke == "1"  ){
                count_n_s += 1;
            }else{
                count_n_n += 1;
            }
        }
    })
    stroke = [
        {
            "name": "已婚",
            "count": count_y_s
        },
        {
            "name": "未婚",
            "count": count_n_s
        },
    ]
    no_stroke = [
        {
            "name": "已婚",
            "count": count_y_n
        },
        {
            "name": "未婚",
            "count": count_n_n
        },
    ]
    for(let i=0;i<stroke.length;i++){
        trace1.x[i] = stroke[i]['name'];
        trace1.y[i] = stroke[i]['count'];
    }

    let trace2 ={};
    trace2.type ="bar";
    trace2.name ="沒中風";
    trace2.x =[];
    trace2.y =[];
    for(let i=0; i<no_stroke.length; i++) {
        trace2.x[i] =no_stroke[i]['name'];
        trace2.y[i] =no_stroke[i]['count'];
    }


        let data =[];
        
        data.push(trace1);
        data.push(trace2);
        let layout ={
            margin:{
                t:0
            }
        };
        Plotly.newPlot(myGraph, data, layout);
    }

function drawSmoke(res) {
    let myGraph = document.getElementById('myGraph2');
    let trace1 ={};
    trace1.type ="bar";
    trace1.name ="中風";
    trace1.x =[];
    trace1.y =[];

    count_n_s = 0;
    count_n_n = 0;

    count_u_s = 0;
    count_u_n = 0;

    count_f_s = 0;
    count_f_n = 0;

    count_s_s = 0;
    count_s_n = 0;

    res.forEach((e) => {
        if(e.smoking_status == 'never smoked'){
            if (e.stroke == "1"  ){
                count_n_s += 1;
            }else{
                count_n_n += 1;
            }
        }else if (e.smoking_status == 'Unknown'){
            if (e.stroke == "1"  ){
                count_u_s += 1;
            }else{
                count_u_n += 1;
            }
        }else if (e.smoking_status == 'formerly smoked'){
            if (e.stroke == "1"  ){
                count_f_s += 1;
            }else{
                count_f_n += 1;
            }
        }else{
            if (e.stroke == "1"  ){
                count_s_s += 1;
            }else{
                count_s_n += 1;
            }
        }


    })
    stroke = [
        {
            "name": "從沒抽過菸",
            "count": count_n_s
        },
        {
            "name": "未知",
            "count": count_u_s
        },
        {
            "name": "之前抽過菸",
            "count": count_f_s
        },
        {
            "name": "還有抽煙",
            "count": count_s_s
        },
    ]
    no_stroke = [
        {
            "name": "從沒抽過菸",
            "count": count_n_n
        },
        {
            "name": "未知",
            "count": count_u_n
        },
        {
            "name": "之前抽過菸",
            "count": count_f_n
        },
        {
            "name": "還有抽煙",
            "count": count_s_n
        },
    ]
    for(let i=0;i<stroke.length;i++){
        trace1.x[i] = stroke[i]['name'];
        trace1.y[i] = stroke[i]['count'];
    }

    let trace2 ={};
    trace2.type ="bar";
    trace2.name ="沒中風";
    trace2.x =[];
    trace2.y =[];
    for(let i=0; i<no_stroke.length; i++) {
        trace2.x[i] =no_stroke[i]['name'];
        trace2.y[i] =no_stroke[i]['count'];
    }


        let data =[];
        
        data.push(trace1);
        data.push(trace2);
        let layout ={
            margin:{
                t:0
            }
        };
        Plotly.newPlot(myGraph, data, layout);
    }