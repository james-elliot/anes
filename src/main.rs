use std::io::Write;

#[allow(non_snake_case)]
#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)]
struct DataCsv {
    N: u64,
    encounterId: u64,
    intervalle: u64,
    heart_rate: Option<f64>,
    temp: Option<f64>,
    pas: Option<f64>,
    fr: Option<f64>,
    spo2: Option<f64>,
    pad: Option<f64>,
    pam: Option<f64>,
}

#[derive(Debug)]
struct Data {
    heart_rate: Option<f64>,
    temp: Option<f64>,
    pas: Option<f64>,
    fr: Option<f64>,
    spo2: Option<f64>,
    pad: Option<f64>,
    pam: Option<f64>,
}

enum KindSelect {
    Global(f64),
    Local(f64),
}

enum KindNormalize {
    Global,
    Local,
    Tainted,
}

fn is_empty_csv(r: &DataCsv) -> bool {
    r.heart_rate.is_none() && r.temp.is_none() && r.pas.is_none()
        && r.fr.is_none() && r.spo2.is_none() && r.pad.is_none() && r.pam.is_none()
}
fn is_empty_data(r: &Data) -> bool {
    r.heart_rate.is_none() && r.temp.is_none() && r.pas.is_none()
        && r.fr.is_none() && r.spo2.is_none() && r.pad.is_none() && r.pam.is_none()
}


use parquet::record::Field;
use parquet::file::serialized_reader::SerializedFileReader;
use parquet::file::reader::FileReader;
fn read_parquet(path: &str) -> Vec<Vec<Data>> {
    {
	let file = std::fs::File::open(path).unwrap();
	let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
	println!("Converted arrow schema is: {}", builder.schema());
    }
    let mut tab = Vec::new();
    let mut prev:Option<u64> = None;
    let mut tabd = Vec::new();
    let mut n = 0;
    let file = std::fs::File::open("one_week.parquet").unwrap();
    let reader = SerializedFileReader::new(file).unwrap();
    let iter = reader.get_row_iter(None).unwrap();
    for row in iter {
        let vect = row.unwrap().into_columns();
//        println!("{:?}",vect);
        let mut r = Data
        {heart_rate: None,temp: None,pas: None,fr: None,spo2: None,pad: None,pam: None};
        for (name,v) in vect {
            match name.as_str() {
                "encounterId" => if let Field::Str(f)=v {
		    let id=Some (f.parse().unwrap());
		    if id != prev {
			if  prev.is_some() {
			    tabd.truncate(tabd.len()-n);
			    tab.push(tabd);
			    tabd = Vec::new();
			    n=0;
			}
			prev=id;
		    }
		},
//                "intervalle" => if let Field::Long(f)=v {r.intervalle=f as u64},
                "heart_rate" => if let Field::Double(f)=v {r.heart_rate=Some(f)},
                "temp" => if let Field::Double(f)=v {r.temp=Some(f)},
                "pas" => if let Field::Double(f)=v {r.pas=Some(f)},
                "fr" => if let Field::Double(f)=v {r.fr=Some(f)},
                "spo2" => if let Field::Double(f)=v {r.spo2=Some(f)},
                "pad" => if let Field::Double(f)=v {r.pad=Some(f)},
                "pam" => if let Field::Double(f)=v {r.pam=Some(f)},
                _ =>{},
            }
        }
	if !tabd.is_empty() || !is_empty_data(&r) {
	    if is_empty_data(&r) {n+=1} else {n=0}
	    tabd.push(r);
	}
    }
    tabd.truncate(tabd.len() - n);
    tab.push(tabd);
    tab
}

fn read_csv(path: &str) -> Vec<Vec<Data>> {
    let mut n2 = 0;
    let mut tab = Vec::new();
    let mut tabd = Vec::new();
    let file = std::fs::File::open(path).unwrap();
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b',')
        .comment(Some(b'#'))
        .has_headers(true)
        .from_reader(file);
    let mut prev = None;
    for result in rdr.deserialize::<DataCsv>() {
        let r = result.unwrap();
        if Some(r.encounterId) != prev {
            if prev.is_some() {
                tabd.truncate(tabd.len() - n2);
                tab.push(tabd);
                tabd = Vec::new();
                n2 = 0;
            }
            prev = Some(r.encounterId);
        }
        if !tabd.is_empty() || !is_empty_csv(&r) {
            let v = Data {
                heart_rate: r.heart_rate,
                temp: r.temp,
                pas: r.pas,
                fr: r.fr,
                spo2: r.spo2,
                pad: r.pad,
                pam: r.pam,
            };
            tabd.push(v);
            if is_empty_csv(&r) {
                n2 += 1;
            } else {
                n2 = 0;
            }
        }
    }
    tabd.truncate(tabd.len() - n2);
    tab.push(tabd);
    tab
}

fn compute(
    tab: &Vec<Data>,f: fn(&Data) -> Option<f64>)
    -> (f64, f64, usize, usize, usize, usize) {
    let (mut sum, mut sum2, mut n) = (0.0, 0.0, 0);
    let mut n2: usize = 0;
    let (mut nmax, mut nbh, mut sumh) = (0, 0, 0);
    for v in tab {
        if let Some(r) = f(v) {
            sum += r; sum2 += r * r; n += 1;
            if n2 != 0 {nbh += 1;sumh += n2;n2 = 0;}
        }
	else {n2 += 1;if n2 > nmax {nmax = n2}}
    }
    if n2 != 0 {nbh += 1;sumh += n2;}
    (sum, sum2, n, nmax, sumh, nbh)
}

fn export(
    tab: &[Data],f: fn(&Data) -> Option<f64>,n: usize,n_min: usize,ind: usize,m: f64,s: f64,
    sou: &mut std::io::BufWriter<std::fs::File>,
    obj: &mut std::io::BufWriter<std::fs::File>) {
    for j in 0..n_min {
        let v = (f(&tab[n - j]).unwrap() - m) / s;
        if j == ind {writeln!(obj, "{}", v).unwrap();} else {write!(sou, "{} ", v).unwrap();}
    }
    writeln!(sou).unwrap();
}

use rand::{Rng, SeedableRng};
pub type Trng = rand_chacha::ChaCha8Rng;

fn compute2(
    tab: &[Data],f: fn(&Data) -> Option<f64>,mt: f64,st: f64,rng: &mut Trng,
    sou_l: &mut std::io::BufWriter<std::fs::File>,
    obj_l: &mut std::io::BufWriter<std::fs::File>,
    sou_t: &mut std::io::BufWriter<std::fs::File>,
    obj_t: &mut std::io::BufWriter<std::fs::File>,
    kind_select: &KindSelect,kind_norm: &KindNormalize,p: f64,n_min: usize,ind: usize) -> u64 {
    let mut n_tot=0;
    let (sou, obj) = if (rng.gen_range(0.0..1.0)) < p {(sou_l, obj_l)} else {(sou_t, obj_t)};
    let (mut sump, mut sump2, mut np) = (0.0, 0.0, 0);
    for v in tab {
        if let Some(r) = f(v) {sump += r; sump2 += r * r; np += 1;}
    }
    let (mp, mp2) = (sump / (np as f64), sump2 / (np as f64));
    let sp = (mp2 - mp * mp).sqrt();
    let mut n3: usize = 0;
    for (i, v) in tab.iter().enumerate() {
        if let Some(r) = f(v) {
            let cond = match kind_select {
                KindSelect::Global(f) => (r - mt).abs() < f * st,
                KindSelect::Local(f) => (r - mp).abs() < f * sp,
            };
            if cond {
                n3 += 1;
                if n3 >= n_min {
		    n_tot+=1;
                    let t = f(&tab[i - ind]).unwrap();
                    let mpl = (sump - t) / ((np - 1) as f64);
                    let mpl2 = (sump2 - t * t) / ((np - 1) as f64);
                    let spl = (mpl2 - mpl * mpl).sqrt();
                    let (mut suml, mut suml2) = (0.0, 0.0);
                    for j in 0..n_min {
                        let x = f(&tab[i - j]).unwrap();
                        suml += x;
                        suml2 += x * x;
                    }
                    let ml = (suml - t) / ((n_min - 1) as f64);
                    let ml2 = (suml2 - t * t) / ((n_min - 1) as f64);
                    let sl = (ml2 - ml * ml).sqrt();
                    match kind_norm {
                        KindNormalize::Global => export(tab, f, i, n_min, ind, mt, st, sou, obj),
                        KindNormalize::Local => export(tab, f, i, n_min, ind, mpl, spl, sou, obj),
                        KindNormalize::Tainted => export(tab, f, i, n_min, ind, ml, sl, sou, obj),
                    }
                }
            }
	    else {n3 = 0;}
        }
	else {n3 = 0;}
    }
    n_tot
}

use std::io::BufRead;
fn read_numbers_from_file(filename: &str) -> std::io::Result<(Vec<f64>,usize,usize)> {
//    let path = std::path::Path::new(filename);
    let file = std::fs::File::open(filename)?;
    let reader = std::io::BufReader::new(file);
    let mut numbers = Vec::new();
    let mut cols = 0;
    let mut rows = 0;
    for line in reader.lines() {
        let line = line?; // Read the line
        let row_numbers: Vec<f64> = line
            .split_whitespace() // Split by whitespace
            .filter_map(|s| s.parse::<f64>().ok()) // Try to parse each number
            .collect();
	cols = row_numbers.len();
        numbers.extend(row_numbers); // Add to the main vector
	rows+=1;
    }
    Ok((numbers,rows,cols))
}

fn to_file(path: &str, tab: &[usize]) {
    let write_file = std::fs::File::create(path).unwrap();
    let mut writer = std::io::BufWriter::new(&write_file);
    for (i, v) in tab.iter().enumerate() {
        if i != 0 {writeln!(&mut writer, "{} {}", i, v).unwrap();}
    }
}

fn update(mut tab: Vec<usize>, v: usize) -> Vec<usize> {
    if (1 + v) > tab.len() {tab.resize(1 + v, 0);}
    tab[v] += 1;
    tab
}

fn get_heart_rate(d: &Data) -> Option<f64> {d.heart_rate}
fn get_temp(d: &Data) -> Option<f64> {d.temp}
fn get_pas(d: &Data) -> Option<f64> {d.pas}
fn get_fr(d: &Data) -> Option<f64> {d.fr}
fn get_spo2(d: &Data) -> Option<f64> {d.spo2}
fn get_pad(d: &Data) -> Option<f64> {d.pad}
fn get_pam(d: &Data) -> Option<f64> {d.pam}

use argparse::{ArgumentParser, Store, StoreFalse, StoreTrue};
use std::collections::HashMap;
use nalgebra::{DMatrix,DVector};
fn main() {
    let mut fmap = HashMap::new();
    fmap.insert("heart_rate", get_heart_rate as fn(&Data) -> Option<f64>);
    fmap.insert("temp", get_temp as fn(&Data) -> Option<f64>);
    fmap.insert("pas", get_pas as fn(&Data) -> Option<f64>);
    fmap.insert("fr", get_fr as fn(&Data) -> Option<f64>);
    fmap.insert("spo2", get_spo2 as fn(&Data) -> Option<f64>);
    fmap.insert("pad", get_pad as fn(&Data) -> Option<f64>);
    fmap.insert("pam", get_pam as fn(&Data) -> Option<f64>);

    let mut fname = "pam".to_string();
    let mut path = "one_week".to_string();
    let mut csv = false;
    let mut fsigma = 2.0;
    let mut proba = 0.8;
    let mut before = 5;
    let mut after = 5;
    let mut seed = 0;
    let mut select_global = true;
    let mut norm_global = false;
    let mut norm_tainted = false;
    {
        // this block limits scope of borrows by ap.refer() method
        let mut ap = ArgumentParser::new();
        ap.set_description("Pre-processing of anes dataset");
        ap.refer(&mut path).add_option(
            &["-f", "--file"],
            Store,
            "Name of the file (without extension) to load (Default: one_week)",
        );
        ap.refer(&mut csv).add_option(
            &["-c", "--csv"],
            StoreTrue,
            "Use csv format instead of parquet format (default: false)",
        );
        ap.refer(&mut fname).add_option(
            &["-n", "--name"],
            Store,
            "Name of the variable to process (Default: pam)",
        );
        ap.refer(&mut proba).add_option(
            &["-p", "--proba"],
            Store,
            "Percentage of the dataset used for learning (Default: 0.8)",
        );
        ap.refer(&mut before).add_option(
            &["-b", "--before"],
            Store,
            "Number of values before the value to predict (Default: 5)",
        );
        ap.refer(&mut after).add_option(
            &["-a", "--after"],
            Store,
            "Number of values after the value to predict (Default: 5)",
        );
        ap.refer(&mut fsigma).add_option(
            &["-f", "--fsigma"],
            Store,
            "Sigma multiplier for select (Default: 2.0)",
        );
        ap.refer(&mut select_global).add_option(
            &["-l", "--LocalSelect"],
            StoreFalse,
            "Use local sigma for select (default: Global)",
        );
        ap.refer(&mut norm_global).add_option(
            &["-g", "--GlobalNormalize"],
            StoreTrue,
            "Use global normalization (default: Local)",
        );
        ap.refer(&mut norm_tainted).add_option(
            &["-t", "--TaintedNormalize"],
            StoreTrue,
            "Use tainted normalization (default: Local)",
        );
        ap.refer(&mut seed).add_option(
            &["-s", "--seed"],
            Store,
            "Seed of random number generator (Default: 0)",
        );
        ap.parse_args_or_exit();
    }
    path = if csv {path+".csv"} else {path+".parquet"};
    let func = if let Some(f)=fmap.get(&*fname) {f} else {panic!("invalid variable name")};
    let kind_select = if select_global {KindSelect::Global(fsigma)}
    else {KindSelect::Local(fsigma)};
    let kind_normalize = if norm_global {
        if norm_tainted {panic!("Can't have both global and tainted")}
        KindNormalize::Global
    }
    else if norm_tainted {KindNormalize::Tainted}
    else {KindNormalize::Local};

    {
	let mut rng = Trng::seed_from_u64(seed);
	let write_file = std::fs::File::create("obj_learn.txt").unwrap();
	let mut obj_l = std::io::BufWriter::new(write_file);
	let write_file = std::fs::File::create("source_learn.txt").unwrap();
	let mut sou_l = std::io::BufWriter::new(write_file);
	let write_file = std::fs::File::create("obj_test.txt").unwrap();
	let mut obj_t = std::io::BufWriter::new(write_file);
	let write_file = std::fs::File::create("source_test.txt").unwrap();
	let mut sou_t = std::io::BufWriter::new(write_file);
	let mut dist_max_hole = Vec::new();
	let mut dist_nb_points = Vec::new();
	let res = if csv {read_csv(&path)} else {read_parquet(&path)};
	let l = res.len();
	let mut sum = 0;
	let (mut sum_all, mut sum2_all, mut n_all) = (0.0, 0.0, 0);
	for v in &res {
            sum += v.len();
	}
	eprintln!(
            "nb_patients={} mean_non_zero_rec_by_patient={}",
            l,
            (sum as f64) / (l as f64)
	);
	for v in &res {
            let (sum, sum2, n, nmax, sumh, nbh) = compute(v, *func);
            let (m, m2) = (sum / (n as f64), sum2 / (n as f64));
            let _s = (m2 - m * m).sqrt();
            let _mh = (sumh as f64) / (nbh as f64);
            sum_all += sum;
            sum2_all += sum2;
            n_all += n;
            //        eprintln!(
            //            "nb={:3} nbnz={:3} m={:5.1} s={:5.1} nmax={:3} nbh={:3} mh={:3.1}",
            //            v.len(),n,m,s,nmax,nbh,mh);
            dist_max_hole = update(dist_max_hole, nmax);
            dist_nb_points = update(dist_nb_points, n);
	}
	to_file("dist_max_hole.txt", &dist_max_hole);
	to_file("dist_nb_points.txt", &dist_nb_points);
	eprintln!(
            "nb_patients_with_holes={} mean_non_zero_rec_by_patient={}",
            l - dist_max_hole[0],
            (n_all as f64) / (l as f64)
	);
	let m = sum_all / (n_all as f64);
	let m2 = sum2_all / (n_all as f64);
	let s = (m2 - m * m).sqrt();
	eprintln!("mean={:?} sigma={:?}", m, s);
	let n_min = before + after + 1;
	let ind = before;
	let mut n_tot = 0;
	for v in &res {
            n_tot += compute2(
		v,*func,m,s,&mut rng,&mut sou_l,&mut obj_l,&mut sou_t,&mut obj_t,
		&kind_select,&kind_normalize,proba,n_min,ind);
	}
	eprintln!("sequences written={}",n_tot);
    }
    // Now let's check files are OK and compute coefs and RMSE for least square method
    let (v1,rows,cols) = read_numbers_from_file("../anes/source_learn.txt").unwrap();
    let a = DMatrix::from_row_slice(rows,cols,&v1);
    let (v2,rows2,cols2) = read_numbers_from_file("../anes/obj_learn.txt").unwrap();
    if cols2!=1 || rows2!=rows {panic!("zorglub")}
    let b = DVector::from_row_slice(&v2);
    let epsilon = 1e-14;
    let results = lstsq::lstsq(&a, &b, epsilon).unwrap();
    let x = results.solution;
    let sum = x.sum();
    eprintln!("coefs: {:?}\nsum: {} RMSE_learn: {}",x.data.as_vec(),sum,(results.residuals/(rows as f64)).sqrt());
    
    let (v1,rows,cols) = read_numbers_from_file("../anes/source_test.txt").unwrap();
    let a = DMatrix::from_row_slice(rows,cols,&v1);
    let (v2,rows2,cols2) = read_numbers_from_file("../anes/obj_test.txt").unwrap();
    if cols2!=1 || rows2!=rows {panic!("zorglub")}
    let b = DVector::from_row_slice(&v2);
    let r = a*x-b;
    let n = r.norm();
    let v = (n*n / (rows as f64)).sqrt();
    eprintln!("RMSE_test: {:?}",v);
}
