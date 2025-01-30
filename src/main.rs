use std::io::Write;
#[allow(non_snake_case)]
#[allow(dead_code)]
#[derive(Debug, serde::Deserialize)]
struct AnesCsv {
    N:u64,
    encounterId:u64,
    intervalle:u64,
    heart_rate:Option<f64>,
    temp:Option<f64>,
    pas:Option<f64>,
    fr:Option<f64>,
    spo2:Option<f64>,
    pad:Option<f64>,
    pam:Option<f64>
}

#[derive(Debug)]
#[allow(dead_code)]
struct Data {
    heart_rate:Option<f64>,
    temp:Option<f64>,
    pas:Option<f64>,
    fr:Option<f64>,
    spo2:Option<f64>,
    pad:Option<f64>,
    pam:Option<f64>
}

fn is_empty(r:&AnesCsv) -> bool {
    r.heart_rate.is_none() && r.temp.is_none() && r.pas.is_none() &&
        r.fr.is_none() && r.spo2.is_none() && r.pad.is_none() && r.pam.is_none()
}

#[allow(dead_code)]
fn read_parquet() {
    let file = std::fs::File::open("one_week.parquet").unwrap();
    let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    println!("Converted arrow schema is: {}", builder.schema());
    let mut reader = builder.build().unwrap();
    let record_batch = reader.next().unwrap().unwrap();
    println!("Read {} records.", record_batch.num_rows());
    let record_batch = reader.next().unwrap().unwrap();
    println!("Read {} records.", record_batch.num_rows());
    println!("{} columns.", record_batch.num_columns());
    let col = record_batch.column_by_name("pam");
    if let Some(_r)=col {
//        let v = std::sync::Arc::get_mut(r.clone());
//        println!("{:?}", v);
    }
}

fn read_anes(path: &str) -> Vec<Vec<Data>> {
    let mut n2 = 0;
    let mut _cpt = 0;
    let mut tab = Vec::new();
    let mut tabd  = Vec::new();
    let file = std::fs::File::open(path).unwrap();
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b',')
        .comment(Some(b'#'))
        .has_headers(true)
        .from_reader(file);
    let mut prev = None ;
    for result in rdr.deserialize::<AnesCsv>() {
        let r = result.unwrap();
        if Some (r.encounterId) != prev {
            if prev.is_some()  {
//                println!("len={:?} n2={:?}",tabd.len(),n2);
                if n2 != 0 {tabd.truncate(tabd.len()-n2);}
                tab.push(tabd);
                tabd=Vec::new();
                _cpt=0;
                n2=0;
            }
            prev = Some (r.encounterId);
        }
        if !tabd.is_empty() || !is_empty(&r) {
            let v= Data {heart_rate:r.heart_rate,temp:r.temp,pas:r.pas,
                         fr:r.fr,spo2:r.spo2,pad:r.pad,pam:r.pam};
//            println!("cpt={:?} n2={:?} v={:?}",cpt,n2,v);
            tabd.push(v);
            if is_empty(&r) {n2+=1;}
            else {n2=0;}
        }
        _cpt+=1;
//        println!("{:?}",r);
    }
    if n2 != 0 {tabd.truncate(tabd.len()-n2);}
    tab.push(tabd);
    tab
}


fn compute(tab:&Vec<Data>,f:fn(&Data)->Option<f64>)
           ->(f64,f64,u64,usize,usize,usize,usize) {
    let (mut sum,mut sum2,mut n) = (0.0,0.0,0);
    let mut n2:usize = 0;
    let (mut nmax,mut nbh,mut sumh) = (0,0,0);
    for v in tab {
        if let Some(r)=f(v) {
            sum+=r;sum2+=r*r;n+=1;
            if n2 != 0 {nbh+=1;sumh+=n2;n2=0;}
        }
        else {n2+=1;if n2>nmax {nmax=n2;}}
    }
    if n2 != 0 {nbh+=1;sumh+=n2;}
    let (m,m2) = (sum/(n as f64),sum2/(n as f64));
    let s = (m2-m*m).sqrt();
    let mut inv = 0;
    for v in tab {if let Some(r)=f(v) {if (r-m).abs()>2.0*s {inv+=1;}}}
    (sum,sum2,n,nmax,inv,sumh,nbh)
}

fn export(tab:&[Data],f:fn(&Data)->Option<f64>,n:usize,n_min:usize,m:f64,s:f64,
          sou:&mut std::io::BufWriter<std::fs::File>,
          obj:&mut std::io::BufWriter<std::fs::File>) {
    for j in 0..n_min {
        let v = (f(&tab[n-j]).unwrap()-m)/s;
        if j==n_min/2 {writeln!(obj,"{}",v).unwrap();}
        else {write!(sou,"{} ",v).unwrap();}
    }
    writeln!(sou).unwrap();
}
use rand::{Rng,SeedableRng};
pub type Trng=rand_chacha::ChaCha8Rng;

fn compute2(tab:&[Data],f:fn(&Data)->Option<f64>,mt:f64,st:f64,rng:&mut Trng,
            sou_l:&mut std::io::BufWriter<std::fs::File>,
            obj_l:&mut std::io::BufWriter<std::fs::File>,
            sou_t:&mut std::io::BufWriter<std::fs::File>,
            obj_t:&mut std::io::BufWriter<std::fs::File>,
            kind:u64) {
    let n_min:usize = 11;
    let (sou,obj)= if (rng.gen_range(0.0..1.0))<0.8 {(sou_l,obj_l)}
    else {(sou_t,obj_t)};
    let (mut sump,mut sump2,mut np) = (0.0,0.0,0);
    for v in tab {if let Some(r)=f(v) {sump+=r;sump2+=r*r;np+=1;}}
    let (mp,mp2) = (sump/(np as f64),sump2/(np as f64));
    let _sp = (mp2-mp*mp).sqrt();
    let mut n3:usize = 0;
    for (i,v) in tab.iter().enumerate() {
        if let Some(r)=f(v) {
            if (r-mp).abs()<2.*_sp {
//            if (r-mt).abs()<2.*st {
                n3 += 1;
                if n3 >= n_min {
                    let k = n_min/2;
                    let t = f(&tab[i-k]).unwrap();
                    let mpl = (sump-t)/((np-1) as f64);
                    let mpl2 = (sump2-t*t)/((np-1) as f64);
                    let spl = (mpl2-mpl*mpl).sqrt();
                    let (mut suml,mut suml2) = (0.0,0.0);
                    for j in 0..n_min {let x=f(&tab[i-j]).unwrap();suml+=x;suml2+=x*x;}
                    let ml = (suml-t)/((n_min-1) as f64);
                    let ml2 = (suml2-t*t)/((n_min-1) as f64);
                    let sl = (ml2-ml*ml).sqrt();
                    if kind==0 {export(tab,f,i,n_min,mt,st,sou,obj);}
                    else if kind==1 {export(tab,f,i,n_min,mpl,spl,sou,obj);}
                    else {export(tab,f,i,n_min,ml,sl,sou,obj);}
                }
            }
            else {n3=0;}
        }
        else {n3=0;}
    }
}

fn to_file(path:&str,tab:&[usize]) {
    let write_file = std::fs::File::create(path).unwrap();
    let mut writer = std::io::BufWriter::new(&write_file);
    for (i,v) in tab.iter().enumerate() {
        if i!=0 {writeln!(&mut writer,"{} {}",i,v).unwrap();}
    }
}
fn update(mut tab:Vec<usize>,v:usize) -> Vec<usize>{
    if (1+v) > tab.len() {tab.resize(1+v,0);}
    tab[v] += 1;
    tab
}
fn main() {
//    read_parquet();
//    std::process::exit(0);
    let mut rng = Trng::seed_from_u64(0);
    let write_file = std::fs::File::create("obj_learn.txt").unwrap();
    let mut obj_l = std::io::BufWriter::new(write_file);
    let write_file = std::fs::File::create("source_learn.txt").unwrap();
    let mut sou_l = std::io::BufWriter::new(write_file);
    let write_file = std::fs::File::create("obj_test.txt").unwrap();
    let mut obj_t = std::io::BufWriter::new(write_file);
    let write_file = std::fs::File::create("source_test.txt").unwrap();
    let mut sou_t = std::io::BufWriter::new(write_file);
    let mut dist_max_hole = Vec::new();
    let mut dist_invalid = Vec::new();
    let mut dist_nb_points = Vec::new();
    let res = read_anes("one_week.csv");
    let l = res.len();
    let mut sum =0;
    let (mut sum_all,mut sum2_all,mut n_all) = (0.0,0.0,0);
    for v in &res {sum+=v.len();}
//    eprintln!("{:?}\n",sum/l);
    for v in &res {
        let (sum,sum2,n,nmax,inv,sumh,nbh) = compute(v,|x| {x.pam});
        let (m,m2) = (sum/(n as f64),sum2/(n as f64));
        let s = (m2-m*m).sqrt();
        sum_all+=sum;
        sum2_all+=sum2;
        n_all+=n;
        let mh = (sumh as f64)/(nbh as f64);
//        eprintln!(
//            "nb={:3} nbnz={:3} m={:5.1} s={:5.1} nmax={:3} inv={:3} nbh={:3} mh={:3.1}",
//            v.len(),n,m,s,nmax,inv,nbh,mh);
        dist_max_hole=update(dist_max_hole,nmax);
        let invp=(100.0*(inv as f64)/(n as f64)) as usize;
        dist_invalid=update(dist_invalid,invp);
        dist_nb_points=update(dist_nb_points,v.len());
        }
    to_file("dist_max_hole.txt",&dist_max_hole);
    to_file("dist_invalid.txt",&dist_invalid);
    to_file("dist_nb_points.txt",&dist_nb_points);
    eprintln!("nbp={:?} nbp_with_holes={:?} nb_samp_mean={:?}",
              l,l-dist_max_hole[0],sum/l);
    let m = sum_all/(n_all as f64);
    let m2 = sum2_all/(n_all as f64);
    let s = (m2-m*m).sqrt();
    eprintln!("mean={:?} sigma={:?}",m,s);
    for v in &res {
        compute2(v,|x| {x.pam},m,s,&mut rng,
                 &mut sou_l,&mut obj_l,&mut sou_t,&mut obj_t,
                 0);
    }
}
